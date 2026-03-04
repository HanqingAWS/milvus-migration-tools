#!/usr/bin/env python3
"""
Milvus Data Export Tool
=======================
Export collection data (schema + index + data) from a Milvus instance to JSON files.
Automatically detects primary key field and data types - no assumptions about data structure.

Usage:
    python3 export_data.py --host 10.0.1.100 --port 19530 --collection my_collection --output ./milvus_export

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


def connect_milvus(host, port, user=None, password=None):
    """Connect to a Milvus instance."""
    kwargs = {"host": host, "port": port}
    if user and password:
        kwargs["user"] = user
        kwargs["password"] = password
    connections.connect("default", **kwargs)
    print(f"Connected to Milvus at {host}:{port}")


def find_primary_field(collection):
    """Auto-detect primary key field from collection schema."""
    for f in collection.schema.fields:
        if f.is_primary:
            return f.name, str(f.dtype)
    return None, None


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


def clean_row(row):
    """Convert numpy types in a row to native Python types for JSON serialization."""
    cleaned = {}
    for k, v in row.items():
        if isinstance(v, np.ndarray):
            cleaned[k] = v.tolist()
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], (np.float32, np.float64)):
            cleaned[k] = [float(x) for x in v]
        elif isinstance(v, (np.float32, np.float64)):
            cleaned[k] = float(v)
        elif isinstance(v, (np.int32, np.int64)):
            cleaned[k] = int(v)
        else:
            cleaned[k] = v
    return cleaned


def export_data_by_pk_iteration(collection, pk_name, pk_dtype, batch_size=500):
    """
    Export all data using primary key based iteration.
    Works with any PK type (INT64, VARCHAR) and any PK values (non-sequential, UUIDs, etc).
    Uses offset/limit pagination with a universal filter expression.
    """
    output_fields = [f.name for f in collection.schema.fields]
    total = collection.num_entities
    all_data = []

    # Use a universal expression that matches all records
    # For INT64 PK: "pk_field >= -9223372036854775808" (INT64 min)
    # For VARCHAR PK: "pk_field != ''" covers non-empty, but we use pk_field > "" for all
    if "INT" in pk_dtype:
        match_all_expr = f'{pk_name} >= -9223372036854775808'
    else:
        # VARCHAR PK - use >= "" to match everything
        match_all_expr = f'{pk_name} != "__IMPOSSIBLE_VALUE_PLACEHOLDER__"'

    for offset in range(0, total, batch_size):
        results = collection.query(
            expr=match_all_expr,
            output_fields=output_fields,
            offset=offset,
            limit=batch_size,
        )
        for row in results:
            all_data.append(clean_row(row))

        exported = min(offset + batch_size, total)
        print(f"  Exported {exported}/{total} records ({exported * 100 // total}%)")

    return all_data


def list_collections_info():
    """List all collections with their basic info."""
    collections = utility.list_collections()
    if not collections:
        print("  (No collections found)")
        return

    print(f"  Found {len(collections)} collection(s):\n")
    print(f"  {'Collection Name':<30} {'Entities':>10}  {'PK Field':<16} {'Vector Field':<16} {'Dim':>5}")
    print(f"  {'-'*30} {'-'*10}  {'-'*16} {'-'*16} {'-'*5}")

    for name in sorted(collections):
        try:
            col = Collection(name)
            count = col.num_entities
            pk_name = "-"
            vec_name = "-"
            dim = "-"
            for f in col.schema.fields:
                if f.is_primary:
                    pk_name = f"{f.name}({str(f.dtype).split('.')[-1]})"
                if "VECTOR" in str(f.dtype):
                    vec_name = f.name
                    dim = str(f.dim) if hasattr(f, "dim") and f.dim else "-"
            print(f"  {name:<30} {count:>10}  {pk_name:<16} {vec_name:<16} {dim:>5}")
        except Exception as e:
            print(f"  {name:<30} {'(error)':>10}  {str(e)[:40]}")

    print()


def export_one_collection(collection_name, output_dir, batch_size=500):
    """Export a single collection. Returns export metadata dict."""
    collection = Collection(collection_name)
    collection.load()

    pk_name, pk_dtype = find_primary_field(collection)
    if not pk_name:
        print(f"  Warning: No primary key found in '{collection_name}', skipping.")
        return None

    print(f"\n  Collection: {collection_name} ({collection.num_entities} entities, PK: {pk_name})")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Schema
    schema_info = export_schema(collection)
    with open(os.path.join(output_dir, "schema.json"), "w", encoding="utf-8") as f:
        json.dump(schema_info, f, indent=2, ensure_ascii=False)
    print(f"  Schema exported: {len(schema_info['fields'])} fields")

    # 2. Index
    index_info = export_index(collection)
    with open(os.path.join(output_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index_info, f, indent=2)
    print(f"  Index exported: {len(index_info)} indexes")

    # 3. Data
    print(f"  Exporting data ({collection.num_entities} records)...")
    export_start = time.time()
    all_data = export_data_by_pk_iteration(collection, pk_name, pk_dtype, batch_size=batch_size)
    export_time = time.time() - export_start

    data_path = os.path.join(output_dir, "data.json")
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, cls=NumpyEncoder, ensure_ascii=False)
    data_size = os.path.getsize(data_path)

    # 4. Metadata
    export_meta = {
        "collection_name": collection_name,
        "total_records": len(all_data),
        "export_time_seconds": round(export_time, 2),
        "data_file_size_bytes": data_size,
        "primary_key_field": pk_name,
        "primary_key_type": pk_dtype,
        "export_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "tool_version": "1.2.0",
    }
    with open(os.path.join(output_dir, "export_meta.json"), "w", encoding="utf-8") as f:
        json.dump(export_meta, f, indent=2)

    print(f"  Done: {len(all_data)} records, {export_time:.2f}s, {data_size / 1024:.1f} KB")
    return export_meta


def main():
    parser = argparse.ArgumentParser(
        description="Export Milvus collection data to JSON files"
    )
    parser.add_argument("--host", default="localhost", help="Milvus host (default: localhost)")
    parser.add_argument("--port", default="19530", help="Milvus port (default: 19530)")
    parser.add_argument("--user", default=None, help="Milvus username (if auth enabled)")
    parser.add_argument("--password", default=None, help="Milvus password (if auth enabled)")
    parser.add_argument("--collection", default=None, help="Collection name to export (omit to use --list or --all)")
    parser.add_argument("--output", default="./milvus_export", help="Output directory (default: ./milvus_export)")
    parser.add_argument("--batch-size", type=int, default=500, help="Query batch size (default: 500)")
    parser.add_argument("--list", action="store_true", help="List all collections and exit")
    parser.add_argument("--all", action="store_true", help="Export all collections")
    args = parser.parse_args()

    print("=" * 60)
    print("Milvus Data Export Tool")
    print("=" * 60)

    # Connect
    connect_milvus(args.host, args.port, args.user, args.password)

    # --list mode: show all collections and exit
    if args.list:
        print("\nCollections on this instance:\n")
        list_collections_info()
        connections.disconnect("default")
        return

    # --all mode: export every collection
    if args.all:
        collections = utility.list_collections()
        if not collections:
            print("\nNo collections found. Nothing to export.")
            connections.disconnect("default")
            return

        print(f"\nExporting all {len(collections)} collection(s)...")
        total_start = time.time()
        results = []

        for name in sorted(collections):
            col_output = os.path.join(args.output, name)
            meta = export_one_collection(name, col_output, batch_size=args.batch_size)
            if meta:
                results.append(meta)

        total_time = time.time() - total_start
        print(f"\n{'=' * 60}")
        print(f"All exports completed!")
        print(f"  Collections: {len(results)}")
        print(f"  Total records: {sum(m['total_records'] for m in results)}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Output: {os.path.abspath(args.output)}/")
        print(f"{'=' * 60}")

        connections.disconnect("default")
        return

    # Single collection mode
    if not args.collection:
        print("\nError: Please specify one of the following:")
        print("  --collection NAME   Export a specific collection")
        print("  --list              List all collections")
        print("  --all               Export all collections")
        print("\nTip: Use --list first to see what collections are available.")
        connections.disconnect("default")
        sys.exit(1)

    # Check collection exists
    if not utility.has_collection(args.collection):
        print(f"\nError: Collection '{args.collection}' does not exist.")
        print(f"\nAvailable collections:")
        list_collections_info()
        sys.exit(1)

    meta = export_one_collection(args.collection, args.output, batch_size=args.batch_size)

    if meta:
        print(f"\n{'=' * 60}")
        print(f"Export completed successfully!")
        print(f"  Records:   {meta['total_records']}")
        print(f"  Time:      {meta['export_time_seconds']}s")
        print(f"  Output:    {os.path.abspath(args.output)}/")
        print(f"{'=' * 60}")

    connections.disconnect("default")


if __name__ == "__main__":
    main()
