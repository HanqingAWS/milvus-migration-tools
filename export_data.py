#!/usr/bin/env python3
"""
Milvus Data Export Tool
=======================
Export collection data (schema + index + data) from a Milvus instance to JSON files.

Usage:
    python3 export_data.py --host 10.0.1.100 --port 19530 --collection game_cs_knowledge --output ./milvus_export

Output:
    <output_dir>/
    ├── schema.json        # Collection schema definition
    ├── index.json         # Index configuration
    ├── data.json          # Full data (including vectors)
    └── export_meta.json   # Export metadata (timestamp, record count, etc.)
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from pymilvus import Collection, connections, utility


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types for JSON serialization."""

    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def connect_milvus(host, port, user=None, password=None, db_name="default"):
    """Connect to a Milvus instance."""
    kwargs = {"host": host, "port": port}
    if user and password:
        kwargs["user"] = user
        kwargs["password"] = password
    connections.connect("default", **kwargs)
    print(f"Connected to Milvus at {host}:{port}")


def export_schema(collection):
    """Export collection schema to a dict."""
    schema_info = {
        "collection_name": collection.name,
        "description": collection.schema.description,
        "fields": [],
    }
    for field in collection.schema.fields:
        field_info = {
            "name": field.name,
            "dtype": str(field.dtype),
            "is_primary": field.is_primary,
            "auto_id": field.auto_id,
        }
        if hasattr(field, "max_length") and field.max_length:
            field_info["max_length"] = field.max_length
        if hasattr(field, "dim") and field.dim:
            field_info["dim"] = field.dim
        schema_info["fields"].append(field_info)
    return schema_info


def export_index(collection):
    """Export index configuration to a list."""
    index_info = []
    for idx in collection.indexes:
        index_info.append(
            {
                "field_name": idx.field_name,
                "index_name": idx.index_name,
                "params": idx.params,
            }
        )
    return index_info


def export_data(collection, batch_size=500):
    """Export all data from a collection."""
    output_fields = [f.name for f in collection.schema.fields]
    total = collection.num_entities
    all_data = []

    for offset in range(0, total, batch_size):
        results = collection.query(
            expr="id >= 0",
            output_fields=output_fields,
            offset=offset,
            limit=batch_size,
        )
        for row in results:
            cleaned = {}
            for k, v in row.items():
                if isinstance(v, (list, np.ndarray)):
                    cleaned[k] = [float(x) for x in v]
                elif isinstance(v, (np.float32, np.float64)):
                    cleaned[k] = float(v)
                elif isinstance(v, (np.int32, np.int64)):
                    cleaned[k] = int(v)
                else:
                    cleaned[k] = v
            all_data.append(cleaned)
        exported = min(offset + batch_size, total)
        print(f"  Exported {exported}/{total} records ({exported * 100 // total}%)")

    return all_data


def main():
    parser = argparse.ArgumentParser(
        description="Export Milvus collection data to JSON files"
    )
    parser.add_argument("--host", default="localhost", help="Milvus host (default: localhost)")
    parser.add_argument("--port", default="19530", help="Milvus port (default: 19530)")
    parser.add_argument("--user", default=None, help="Milvus username (if auth enabled)")
    parser.add_argument("--password", default=None, help="Milvus password (if auth enabled)")
    parser.add_argument("--collection", required=True, help="Collection name to export")
    parser.add_argument("--output", default="./milvus_export", help="Output directory (default: ./milvus_export)")
    parser.add_argument("--batch-size", type=int, default=500, help="Query batch size (default: 500)")
    args = parser.parse_args()

    print("=" * 60)
    print("Milvus Data Export Tool")
    print("=" * 60)

    # Connect
    connect_milvus(args.host, args.port, args.user, args.password)

    # Check collection
    if not utility.has_collection(args.collection):
        print(f"Error: Collection '{args.collection}' does not exist.")
        print(f"Available collections: {utility.list_collections()}")
        sys.exit(1)

    collection = Collection(args.collection)
    collection.load()
    print(f"Collection: {args.collection} ({collection.num_entities} entities)")

    # Create output dir
    os.makedirs(args.output, exist_ok=True)

    # 1. Export schema
    print("\n[1/3] Exporting schema...")
    schema_info = export_schema(collection)
    with open(os.path.join(args.output, "schema.json"), "w", encoding="utf-8") as f:
        json.dump(schema_info, f, indent=2, ensure_ascii=False)
    print(f"  Schema exported: {len(schema_info['fields'])} fields")

    # 2. Export index
    print("\n[2/3] Exporting index configuration...")
    index_info = export_index(collection)
    with open(os.path.join(args.output, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index_info, f, indent=2)
    print(f"  Index exported: {len(index_info)} indexes")

    # 3. Export data
    print(f"\n[3/3] Exporting data ({collection.num_entities} records)...")
    export_start = time.time()
    all_data = export_data(collection, batch_size=args.batch_size)
    export_time = time.time() - export_start

    data_path = os.path.join(args.output, "data.json")
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, cls=NumpyEncoder, ensure_ascii=False)
    data_size = os.path.getsize(data_path)

    # 4. Save metadata
    export_meta = {
        "collection_name": args.collection,
        "total_records": len(all_data),
        "export_time_seconds": round(export_time, 2),
        "data_file_size_bytes": data_size,
        "source_host": args.host,
        "source_port": args.port,
        "export_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "tool_version": "1.0.0",
    }
    with open(os.path.join(args.output, "export_meta.json"), "w", encoding="utf-8") as f:
        json.dump(export_meta, f, indent=2)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Export completed successfully!")
    print(f"  Records:   {len(all_data)}")
    print(f"  Time:      {export_time:.2f}s")
    print(f"  Data size: {data_size / 1024:.1f} KB")
    print(f"  Output:    {os.path.abspath(args.output)}/")
    print(f"{'=' * 60}")

    connections.disconnect("default")


if __name__ == "__main__":
    main()
