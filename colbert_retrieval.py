"""
ColBERT-based Dense Retrieval System for Legal Precedents
This implementation creates a dense retrieval system using ColBERT for legal precedent retrieval.

Requirements:
- PyTorch
- Transformers
- Pyserini (for comparison with BM25)
- faiss-gpu (for efficient similarity search)
"""

import os
import json
import torch
import numpy as np
import argparse
import faiss
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Union, Tuple
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader

# Path configuration
def get_base_dir() -> Path:
    env = os.getenv("CASE_JSON_DIR")
    if env:
        return Path(env)
    project_root = Path(__file__).resolve().parents[1]
    default = project_root / "data" / "Caselaw_Pennsylvania_State_Reports_1845-2017"
    if not default.exists():
        raise FileNotFoundError(
            f"Could not find default data directory at {default}. "
            "Set CASE_JSON_DIR or pass --data-dir explicitly."
        )
    return default

class LegalCaseDataset(Dataset):
    """Dataset for loading and preprocessing legal cases"""
    
    def __init__(self, data_dir: Path, max_length: int = 512):
        self.data_dir = data_dir
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Load all case IDs and paths
        self.cases = []
        for json_file in tqdm(list(data_dir.rglob("**/json/*.json")), desc="Scanning case files"):
            with open(json_file, "r") as f:
                try:
                    data = json.load(f)
                    case_id = str(data.get("id"))
                    self.cases.append((case_id, json_file))
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse {json_file}")
        
        print(f"Loaded {len(self.cases)} cases")
    
    def __len__(self):
        return len(self.cases)
    
    def __getitem__(self, idx):
        case_id, file_path = self.cases[idx]
        
        with open(file_path, "r") as f:
            case_data = json.load(f)
        
        # Construct contents similar to the BM25 approach
        contents = []
        
        # Add decision date if available
        if "decision_date" in case_data:
            contents.append(f"Decision Date: {case_data['decision_date']}")
        
        # Add case name if available
        if "name" in case_data:
            contents.append(f"Case Name: {case_data['name']}")
        
        # Add name abbreviation if available
        if "name_abbreviation" in case_data:
            contents.append(f"Name Abbreviation: {case_data['name_abbreviation']}")
        
        # Add opinions text if available
        if "opinions" in case_data:
            for opinion in case_data["opinions"]:
                if "text" in opinion:
                    contents.append(opinion["text"])
        
        # Join all content
        text = " ".join(contents)
        
        # Tokenize and truncate
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "case_id": case_id,
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "text": text
        }

class ColBERTEncoder:
    """Implements ColBERT encoding for legal documents and queries"""
    
    def __init__(self, model_name: str = "bert-base-uncased", device: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Using device: {self.device}")
    
    def encode_batch(self, batch, max_length: int = 512):
        """Encode a batch of texts using the ColBERT approach"""
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # Get the last hidden state
            last_hidden_state = outputs.last_hidden_state
            
            # Apply mask to get only valid token representations
            masked_embeddings = last_hidden_state * attention_mask.unsqueeze(-1)
            
            # Normalize each token embedding (L2 norm)
            embeddings_norm = torch.norm(masked_embeddings, p=2, dim=2, keepdim=True)
            normalized_embeddings = masked_embeddings / (embeddings_norm + 1e-9)
            
            # Return the token-level embeddings along with the mask
            return normalized_embeddings, attention_mask
    
    def encode_query(self, query: str, max_length: int = 64):
        """Encode a query using the ColBERT approach"""
        encoding = self.tokenizer(
            query,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # Get the last hidden state
            last_hidden_state = outputs.last_hidden_state
            
            # Apply mask
            masked_embeddings = last_hidden_state * attention_mask.unsqueeze(-1)
            
            # Normalize
            embeddings_norm = torch.norm(masked_embeddings, p=2, dim=2, keepdim=True)
            normalized_embeddings = masked_embeddings / (embeddings_norm + 1e-9)
            
            return normalized_embeddings, attention_mask

class LegalColBERTRetrieval:
    """Main retrieval system using ColBERT for legal precedents"""
    
    def __init__(
        self,
        data_dir: Path,
        max_length: int = 512,
        batch_size: int = 8,
        model_name: str = "bert-base-uncased",
        index_name: str = "colbert_index",
        device: str = None
    ):
        self.data_dir = data_dir
        self.max_length = max_length
        self.batch_size = batch_size
        self.model_name = model_name
        self.index_name = index_name
        
        # Initialize dataset and encoder
        self.dataset = LegalCaseDataset(data_dir, max_length)
        self.encoder = ColBERTEncoder(model_name, device)
        
        # Case ID mapping
        self.id_to_index = {case_id: i for i, (case_id, _) in enumerate(self.dataset.cases)}
        self.index_to_id = {i: case_id for case_id, i in self.id_to_index.items()}
        
        # Check if index exists
        self.index_path = Path(f"{index_name}.faiss")
        self.embeddings_path = Path(f"{index_name}_embeddings.npy")
        self.masks_path = Path(f"{index_name}_masks.npy")
        
        self.faiss_index = None
        self.document_embeddings = None
        self.document_masks = None
    
    def build_index(self, force_rebuild: bool = False):
        """Build or load the FAISS index for document embeddings"""
        if not force_rebuild and self.index_path.exists() and self.embeddings_path.exists() and self.masks_path.exists():
            print(f"Loading existing index from {self.index_path}")
            self.faiss_index = faiss.read_index(str(self.index_path))
            self.document_embeddings = np.load(str(self.embeddings_path), allow_pickle=True)
            self.document_masks = np.load(str(self.masks_path), allow_pickle=True)
            print(f"Loaded index with {self.faiss_index.ntotal} documents")
            return
        
        print("Building new index...")
        
        # Create dataloader
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        # Encode all documents
        all_embeddings = []
        all_masks = []
        all_ids = []
        
        for batch in tqdm(dataloader, desc="Encoding documents"):
            case_ids = batch["case_id"]
            embeddings, masks = self.encoder.encode_batch(batch)
            
            # Move to CPU and convert to numpy
            embeddings_cpu = embeddings.cpu().numpy()
            masks_cpu = masks.cpu().numpy()
            
            for i, case_id in enumerate(case_ids):
                all_embeddings.append(embeddings_cpu[i])
                all_masks.append(masks_cpu[i])
                all_ids.append(case_id)
        
        # Save embeddings and masks
        self.document_embeddings = np.array(all_embeddings, dtype=object)
        self.document_masks = np.array(all_masks, dtype=object)
        
        # Build FAISS index for first token ([CLS]) embeddings for initial retrieval
        cls_embeddings = np.array([emb[0] for emb in all_embeddings])
        dim = cls_embeddings.shape[1]
        
        # Create index
        self.faiss_index = faiss.IndexFlatIP(dim)  # Inner product (cosine similarity for normalized vectors)
        self.faiss_index.add(cls_embeddings)
        
        # Save the index and embeddings
        faiss.write_index(self.faiss_index, str(self.index_path))
        np.save(str(self.embeddings_path), self.document_embeddings)
        np.save(str(self.masks_path), self.document_masks)
        
        # Update ID mappings
        self.id_to_index = {case_id: i for i, case_id in enumerate(all_ids)}
        self.index_to_id = {i: case_id for case_id, i in self.id_to_index.items()}
        
        print(f"Built index with {self.faiss_index.ntotal} documents")
    
    def retrieve(self, query: str, k: int = 100, rerank_k: int = 10) -> List[Dict]:
        """
        Two-stage retrieval:
        1. Use FAISS to retrieve k candidates based on CLS embeddings
        2. Rerank top k using full ColBERT MaxSim scoring
        """
        # Encode query
        query_embeddings, query_mask = self.encoder.encode_query(query)
        query_embeddings_np = query_embeddings.cpu().numpy()
        query_mask_np = query_mask.cpu().numpy()
        
        # First-stage retrieval using FAISS
        cls_query = query_embeddings_np[0, 0].reshape(1, -1)  # Use CLS token
        scores, indices = self.faiss_index.search(cls_query, k)
        
        candidate_indices = indices[0]
        
        # Second-stage reranking using ColBERT MaxSim
        results = []
        
        for idx in candidate_indices:
            case_id = self.index_to_id[idx]
            doc_embedding = self.document_embeddings[idx]
            doc_mask = self.document_masks[idx]
            
            # Compute MaxSim score between query and document tokens
            similarity_matrix = np.matmul(
                query_embeddings_np[0],  # Query tokens
                doc_embedding.T  # Document tokens
            )
            
            # Mask out padding tokens
            valid_query_tokens = query_mask_np[0].sum()
            valid_doc_tokens = doc_mask.sum()
            
            masked_similarity = similarity_matrix[:valid_query_tokens, :valid_doc_tokens]
            
            # MaxSim operation: for each query token, find the maximum similarity with any document token
            max_sim_per_query_token = np.max(masked_similarity, axis=1)
            
            # Sum the MaxSim values
            score = np.sum(max_sim_per_query_token)
            
            # Normalize by query length
            score = score / valid_query_tokens if valid_query_tokens > 0 else 0
            
            # Get case details
            case_details = self._get_case_details(case_id)
            
            results.append({
                "case_id": case_id,
                "score": float(score),
                "details": case_details
            })
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top rerank_k results
        return results[:rerank_k]
    
    def _get_case_details(self, case_id: str) -> Dict:
        """Retrieve case details from the original JSON file"""
        for id, file_path in self.dataset.cases:
            if id == case_id:
                with open(file_path, "r") as f:
                    data = json.load(f)
                
                return {
                    "id": case_id,
                    "name": data.get("name", ""),
                    "name_abbreviation": data.get("name_abbreviation", ""),
                    "decision_date": data.get("decision_date", ""),
                    "court": data.get("court", {}).get("name", ""),
                    "docket_number": data.get("docket_number", ""),
                }
        
        return {
            "id": case_id,
            "name": "Unknown",
            "name_abbreviation": "Unknown",
            "decision_date": "",
            "court": "",
            "docket_number": "",
        }
    
    def evaluate(self, case_id: str, k_values: List[int] = [5, 10, 20, 50]) -> Dict:
        """Evaluate retrieval for a specific case using citation network as gold labels"""
        # Get gold labels (cited cases)
        gold_labels = get_gold_labels(case_id, self.data_dir)
        
        if not gold_labels:
            print(f"Warning: No gold labels found for case {case_id}")
            return {}
        
        # Get case text for query
        case_index = self.id_to_index.get(case_id)
        if case_index is None:
            print(f"Warning: Case {case_id} not found in index")
            return {}
        
        case_data = self.dataset[case_index]
        query_text = case_data["text"]
        
        # Truncate query if too long
        if len(query_text) > 1024:
            query_text = query_text[:1024]
        
        # Retrieve cases
        max_k = max(k_values)
        retrieved_cases = self.retrieve(query_text, k=max_k * 2, rerank_k=max_k)
        retrieved_ids = [case["case_id"] for case in retrieved_cases]
        
        # Calculate metrics
        metrics = {}
        
        # Precision@k and Recall@k
        for k in k_values:
            top_k_ids = retrieved_ids[:k]
            relevant_retrieved = set(top_k_ids).intersection(set(gold_labels))
            
            precision = len(relevant_retrieved) / k if k > 0 else 0
            recall = len(relevant_retrieved) / len(gold_labels) if gold_labels else 0
            
            metrics[f"precision@{k}"] = precision
            metrics[f"recall@{k}"] = recall
        
        # Mean Reciprocal Rank (MRR)
        reciprocal_ranks = []
        for gold_id in gold_labels:
            if gold_id in retrieved_ids:
                rank = retrieved_ids.index(gold_id) + 1
                reciprocal_ranks.append(1 / rank)
            else:
                reciprocal_ranks.append(0)
        
        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0
        metrics["mrr"] = mrr
        
        # Mean Average Precision (MAP)
        avg_precision = 0
        relevant_count = 0
        
        for i, case_id in enumerate(retrieved_ids[:max_k]):
            if case_id in gold_labels:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                avg_precision += precision_at_i
        
        map_score = avg_precision / len(gold_labels) if gold_labels else 0
        metrics["map"] = map_score
        
        return metrics

def get_gold_labels(
    case_id: Union[int, str],
    base_dir: Path = None,
) -> List[str]:
    """Get gold labels (cited cases) for a specific case"""
    if base_dir is None:
        base_dir = get_base_dir()
    
    data = None
    # Search for the case JSON file
    for json_file in base_dir.rglob("**/json/*.json"):
        with open(json_file, "r") as f:
            try:
                d = json.load(f)
                if str(d.get("id")) == str(case_id):
                    data = d
                    break
            except json.JSONDecodeError:
                continue
    
    if data is None:
        return []
    
    # Extract numeric case_ids from citations
    results = []
    for entry in data.get("cites_to", []):
        for cid in entry.get("case_ids", []):
            results.append(str(cid))
    
    return sorted(set(results))

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="ColBERT-based Dense Retrieval for Legal Precedents"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing case JSON files"
    )
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build or rebuild the index"
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuilding the index even if it exists"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Query text for retrieval"
    )
    parser.add_argument(
        "--case-id",
        type=str,
        help="Case ID to use for evaluation or as query"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to return"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate retrieval using citation network as gold labels"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for encoding"
    )
    parser.add_argument(
        "--index-name",
        type=str,
        default="colbert_index",
        help="Name for the FAISS index files"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda or cpu)"
    )
    
    return parser.parse_args()

def load_retrieval(
    data_dir=None,
    index_name="colbert_index",
    batch_size=8,
    device=None,
    force_rebuild=False,
):
    """Convenience wrapper so other scripts can do retr = load_retrieval(device="cuda")"""
    data_dir = Path(data_dir) if data_dir else get_base_dir()
    retr = LegalColBERTRetrieval(
        data_dir=data_dir,
        batch_size=batch_size,
        index_name=index_name,
        device=device,
    )
    retr.build_index(force_rebuild=force_rebuild)
    return retr

def main():
    """Main function"""
    args = parse_args()
    
    # Set up data directory
    data_dir = args.data_dir if args.data_dir is not None else get_base_dir()
    
    print(f"Using data directory: {data_dir}")
    
    # Create retrieval system
    retrieval = LegalColBERTRetrieval(
        data_dir=data_dir,
        batch_size=args.batch_size,
        index_name=args.index_name,
        device=args.device
    )
    
    # Build index if requested
    if args.build_index or args.force_rebuild:
        retrieval.build_index(force_rebuild=args.force_rebuild)
    else:
        # Load existing index
        retrieval.build_index(force_rebuild=False)
    
    # Process query if provided
    if args.query:
        results = retrieval.retrieve(args.query, k=100, rerank_k=args.top_k)
        
        print(f"\nTop {len(results)} results for query: '{args.query}'")
        print("=" * 80)
        
        for i, result in enumerate(results):
            print(f"{i+1}. {result['details']['name_abbreviation']} ({result['details']['decision_date']})")
            print(f"   ID: {result['case_id']}")
            print(f"   Score: {result['score']:.4f}")
            print(f"   Court: {result['details']['court']}")
            if result['details']['docket_number']:
                print(f"   Docket: {result['details']['docket_number']}")
            print()
    
    # Process case ID if provided
    if args.case_id:
        if args.evaluate:
            # Evaluate retrieval for this case
            metrics = retrieval.evaluate(args.case_id)
            
            print(f"\nEvaluation metrics for case {args.case_id}:")
            print("=" * 80)
            
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
        else:
            # Use case as query
            case_index = retrieval.id_to_index.get(args.case_id)
            if case_index is None:
                print(f"Error: Case {args.case_id} not found")
                return
            
            case_data = retrieval.dataset[case_index]
            query_text = case_data["text"]
            
            # Truncate query if too long
            if len(query_text) > 1024:
                query_text = query_text[:1024]
            
            results = retrieval.retrieve(query_text, k=100, rerank_k=args.top_k)
            
            print(f"\nTop {len(results)} results for case {args.case_id}:")
            print("=" * 80)
            
            for i, result in enumerate(results):
                print(f"{i+1}. {result['details']['name_abbreviation']} ({result['details']['decision_date']})")
                print(f"   ID: {result['case_id']}")
                print(f"   Score: {result['score']:.4f}")
                print(f"   Court: {result['details']['court']}")
                if result['details']['docket_number']:
                    print(f"   Docket: {result['details']['docket_number']}")
                print()

if __name__ == "__main__":
    main()
