#!/usr/bin/env python3
# load_test.py
# A simple load testing script for the inference worker

import asyncio
import aiohttp
import argparse
import json
import time
import uuid
import statistics
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

console = Console()

# Default configuration
DEFAULT_WORKER_URL = "http://localhost:3000"
DEFAULT_API_KEY = "fart"
DEFAULT_MODEL_ID = "bge-small-en-v1.5"
DEFAULT_BATCH_SIZE = 10  # Documents per batch
DEFAULT_NUM_BATCHES = 10
DEFAULT_CONCURRENCY = 3
DEFAULT_DOCUMENT_SIZE = 100  # Token count

# Generate a test document
def generate_test_document(token_count):
    words = ["embedding", "vector", "semantic", "search", "neural", 
            "network", "machine", "learning", "artificial", "intelligence",
            "natural", "language", "processing", "transformer", "bert", 
            "model", "inference", "worker", "microservice", "gpu"]
    
    # Generate some text with approximately token_count tokens
    word_count = max(10, token_count // 2)  # Rough approximation of tokens to words
    text = " ".join([words[i % len(words)] for i in range(word_count)])
    
    return {
        "service_id": f"test-service-{uuid.uuid4()}",
        "tokenized_text": text,
        "token_count": token_count,
        "job_id": str(uuid.uuid4())
    }

# Create a batch of test documents
def create_test_batch(batch_size, doc_size, model_id):
    return {
        "documents": [generate_test_document(doc_size) for _ in range(batch_size)],
        "request_id": str(uuid.uuid4()),
        "priority": 1,
        "model_id": model_id
    }

# Process a single batch and return the result and timing
async def process_batch(session, worker_url, api_key, batch, timeout=60):
    start_time = time.time()
    try:
        async with session.post(
            f"{worker_url}/api/batches",
            headers={"X-API-Key": api_key, "Content-Type": "application/json"},
            json=batch,
            timeout=timeout
        ) as response:
            elapsed = time.time() - start_time
            response_json = await response.json()
            return {
                "success": response.status == 200,
                "status": response.status,
                "elapsed_time": elapsed,
                "server_time": response_json.get("processing_time_ms", 0) / 1000,
                "document_count": len(batch["documents"]),
                "total_tokens": sum(doc["token_count"] for doc in batch["documents"]),
                "embedding_dim": len(response_json.get("results", [{}])[0].get("embedding", [])) if response.status == 200 and "results" in response_json and response_json["results"] else 0,
                "error": response_json.get("error", None),
            }
    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "status": "Timeout",
            "elapsed_time": elapsed,
            "server_time": 0,
            "document_count": len(batch["documents"]),
            "total_tokens": sum(doc["token_count"] for doc in batch["documents"]),
            "embedding_dim": 0,
            "error": "Request timed out",
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "status": "Error",
            "elapsed_time": elapsed,
            "server_time": 0,
            "document_count": len(batch["documents"]),
            "total_tokens": sum(doc["token_count"] for doc in batch["documents"]),
            "embedding_dim": 0,
            "error": str(e),
        }

# Get worker status
async def get_worker_status(session, worker_url, api_key):
    try:
        async with session.get(
            f"{worker_url}/api/status",
            headers={"X-API-Key": api_key},
            timeout=10
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                return {"error": f"Failed to get status: {response.status}"}
    except Exception as e:
        return {"error": str(e)}

# Get GPU metrics
async def get_gpu_metrics(session, worker_url, api_key):
    try:
        async with session.get(
            f"{worker_url}/api/gpu/memory",
            headers={"X-API-Key": api_key},
            timeout=10
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                return {"error": f"Failed to get GPU metrics: {response.status}"}
    except Exception as e:
        return {"error": str(e)}

# Main load test function
async def run_load_test(args):
    console.print(f"[bold cyan]Starting load test against {args.worker_url}[/bold cyan]")
    console.print(f"Configuration: {args.num_batches} batches, {args.batch_size} documents per batch, {args.concurrency} concurrent requests")
    
    # Create session
    async with aiohttp.ClientSession() as session:
        # Check worker status before starting
        console.print("[yellow]Checking worker status...[/yellow]")
        status = await get_worker_status(session, args.worker_url, args.api_key)
        
        if "error" in status:
            console.print(f"[bold red]Error getting worker status: {status['error']}[/bold red]")
            return
        
        console.print("[green]Worker status:[/green]")
        console.print(f"  Worker ID: {status.get('worker_id', 'unknown')}")
        console.print(f"  Status: {status.get('status', 'unknown')}")
        console.print(f"  GPU: {status.get('capabilities', {}).get('gpu_type', 'unknown')}")
        console.print(f"  GPU Memory: {status.get('capabilities', {}).get('gpu_memory_mb', 'unknown')} MB")
        console.print(f"  Optimal Batch Size: {status.get('capabilities', {}).get('optimal_batch_size', 'unknown')}")
        
        # Generate test batches
        batches = [create_test_batch(args.batch_size, args.document_size, args.model_id) for _ in range(args.num_batches)]
        
        # Process batches with concurrency limit
        results = []
        
        with Progress() as progress:
            task = progress.add_task("[cyan]Processing batches...", total=len(batches))
            
            for i in range(0, len(batches), args.concurrency):
                batch_slice = batches[i:i+args.concurrency]
                tasks = [process_batch(session, args.worker_url, args.api_key, batch, args.timeout) for batch in batch_slice]
                batch_results = await asyncio.gather(*tasks)
                results.extend(batch_results)
                progress.update(task, advance=len(batch_slice))
                
                # Small delay between batch groups to allow GPU to recover
                if i + args.concurrency < len(batches):
                    await asyncio.sleep(1)
        
        # Get final GPU metrics
        gpu_metrics = await get_gpu_metrics(session, args.worker_url, args.api_key)
        
        # Analyze results
        successful_requests = [r for r in results if r["success"]]
        failed_requests = [r for r in results if not r["success"]]
        
        # Create results table
        table = Table(title="Load Test Results")
        
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Requests", str(len(results)))
        table.add_row("Successful Requests", str(len(successful_requests)))
        table.add_row("Failed Requests", str(len(failed_requests)))
        
        if successful_requests:
            avg_time = statistics.mean([r["elapsed_time"] for r in successful_requests])
            max_time = max([r["elapsed_time"] for r in successful_requests])
            min_time = min([r["elapsed_time"] for r in successful_requests])
            avg_server_time = statistics.mean([r["server_time"] for r in successful_requests])
            
            table.add_row("Avg Response Time", f"{avg_time:.3f}s")
            table.add_row("Min Response Time", f"{min_time:.3f}s")
            table.add_row("Max Response Time", f"{max_time:.3f}s")
            table.add_row("Avg Server Processing Time", f"{avg_server_time:.3f}s")
            
            # Calculate throughput
            total_docs = sum([r["document_count"] for r in successful_requests])
            total_tokens = sum([r["total_tokens"] for r in successful_requests])
            total_time = sum([r["elapsed_time"] for r in successful_requests])
            
            table.add_row("Total Documents Processed", str(total_docs))
            table.add_row("Total Tokens Processed", str(total_tokens))
            table.add_row("Documents per Second", f"{total_docs/total_time:.2f}")
            table.add_row("Tokens per Second", f"{total_tokens/total_time:.2f}")
            
            # Embedding dimensions
            if successful_requests[0]["embedding_dim"] > 0:
                table.add_row("Embedding Dimensions", str(successful_requests[0]["embedding_dim"]))
        
        # GPU metrics
        if "error" not in gpu_metrics:
            table.add_row("Final GPU Memory Used", f"{gpu_metrics.get('used_mb', 'N/A')} MB")
            table.add_row("Final GPU Memory Free", f"{gpu_metrics.get('free_mb', 'N/A')} MB")
            table.add_row("GPU Memory Utilization", f"{gpu_metrics.get('utilization_percent', 'N/A')}%")
        
        console.print(table)
        
        # Print errors if any
        if failed_requests:
            console.print("[bold red]Errors:[/bold red]")
            for i, req in enumerate(failed_requests):
                console.print(f"  {i+1}. Status: {req['status']}, Error: {req['error']}")

# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Load test for Inference Worker")
    parser.add_argument("--worker-url", default=DEFAULT_WORKER_URL, help="Inference worker URL")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API key for authentication")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Model ID to use")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Number of documents per batch")
    parser.add_argument("--num-batches", type=int, default=DEFAULT_NUM_BATCHES, help="Total number of batches to process")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Number of concurrent requests")
    parser.add_argument("--document-size", type=int, default=DEFAULT_DOCUMENT_SIZE, help="Token count per document")
    parser.add_argument("--timeout", type=int, default=60, help="Request timeout in seconds")
    return parser.parse_args()

# Main entry point
if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_load_test(args))