#!/usr/bin/env python3
"""
Milvus Data Import Tool
=======================
Import collection data from JSON files (exported by export_data.py) into a Milvus instance.

Usage:
    python3 import_data.py --host 10.0.2.200 --port 19530 --input ./milvus_export

Input:
    <input_dir>/
    ├── schema.json    # Collection schema definition
    ├── index.json     # Index configuration
    └── data.json      # Full data (including vectors)
"""

import argparse
import json
import os
import sys
import time

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

# Mapping from string dtype to pymilvus DataType
DTYPE_MAP = {
    "DataType.BOOL": DataType.BOOL,
    "DataType.INT8": DataType.INT8,
    "DataType.INT16": DataType.INT16,
    "DataType.INT32": DataType.INT32,
    "DataType.INT64": DataType.INT64,
    "DataType.FLOAT": DataType.FLOAT,
    "DataType.DOUBLE": DataType.DOUBLE,
    "DataType.VARCHAR": DataType.VARCHAR,
    "DataType.JSON": DataType.JSON,
    "DataType.FLOAT_VECTOR": DataType.FLOAT_VECTOR,
    "DataType.BINARY_VECTOR": DataType.BINARY_VECTOR,
}


def connect_milvus(host, port, user=None, password=None):
    """Connect to a Milvus instance."""
    kwargs = {"host": host, "port": port}
    if user and password:
        kwargs["user"] = user
        kwargs["password"] = password
    connections.connect("default", **kwargs)
    print(f"Connected to Milvus at {host}:{port}")


def load_export_files(input_dir):
    """Load schema, index, and data files from export directory."""
    schema_path = os.path.join(input_dir, "schema.json")
    index_path = os.path.join(input_dir, "index.json")
    data_path = os.path.join(input_dir, "data.json")

    for path, name in [(schema_path, "schema.json"), (data_path, "data.json")]:
        if not os.path.exists(path):
            print(f"Error: {name} not found in {input_dir}")
            sys.exit(1)

    with open(schema_path, encoding="utf-8") as f:
        schema_info = json.load(f)
    with open(index_path, encoding="utf-8") as f:
        index_info = json.load(f)

    print(f"Loading data file: {data_path}")
    with open(data_path, encoding="utf-8") as f:
        all_data = json.load(f)
    print(f"  Loaded {len(all_data)} records")

    return schema_info, index_info, all_data


def create_collection(schema_info, drop_existing=False):
    """Create collection from schema definition."""
    collection_name = schema_info["collection_name"]

    if utility.has_collection(collection_name):
        if drop_existing:
            utility.drop_collection(collection_name)
            print(f"  Dropped existing collection '{collection_name}'")
        else:
            print(f"Error: Collection '{collection_name}' already exists.")
            print(f"  Use --drop-existing to drop it before importing.")
            sys.exit(1)

    fields = []
    for fi in schema_info["fields"]:
        if fi["dtype"] not in DTYPE_MAP:
            print(f"Error: Unsupported dtype '{fi['dtype']}' for field '{fi['name']}'")
            sys.exit(1)

        kwargs = {}
        if fi.get("max_length"):
            kwargs["max_length"] = fi["max_length"]
        if fi.get("dim"):
            kwargs["dim"] = fi["dim"]
        if fi.get("auto_id"):
            kwargs["auto_id"] = True

        fields.append(
            FieldSchema(
                name=fi["name"],
                dtype=DTYPE_MAP[fi["dtype"]],
                is_primary=fi["is_primary"],
                **kwargs,
            )
        )

    schema = CollectionSchema(fields, description=schema_info.get("description", ""))
    collection = Collection(collection_name, schema)
    print(f"  Collection '{collection_name}' created ({len(fields)} fields)")
    return collection


def insert_data(collection, schema_info, all_data, batch_size=200):
    """Insert data into collection in batches."""
    field_names = [f["name"] for f in schema_info["fields"]]
    total = len(all_data)
    total_inserted = 0

    for i in range(0, total, batch_size):
        batch = all_data[i : i + batch_size]
        batch_data = [[row[fn] for row in batch] for fn in field_names]
        collection.insert(batch_data)
        total_inserted += len(batch)
        print(f"  Imported {total_inserted}/{total} records ({total_inserted * 100 // total}%)")

    collection.flush()
    return total_inserted


def create_indexes(collection, index_info):
    """Recreate indexes from index configuration."""
    for idx in index_info:
        params = idx["params"]
        index_params = {
            "metric_type": params.get("metric_type", "COSINE"),
            "index_type": params.get("index_type", "IVF_FLAT"),
        }
        # Extract nested params (like nlist, nprobe, etc.)
        inner_params = params.get("params", {})
        if isinstance(inner_params, dict) and inner_params:
            index_params["params"] = inner_params
        else:
            index_params["params"] = {"nlist": 64}

        collection.create_index(idx["field_name"], index_params)
        print(f"  Index created on '{idx['field_name']}': {index_params['index_type']} / {index_params['metric_type']}")


def main():
    parser = argparse.ArgumentParser(
        description="Import Milvus collection data from JSON files"
    )
    parser.add_argument("--host", default="localhost", help="Milvus host (default: localhost)")
    parser.add_argument("--port", default="19530", help="Milvus port (default: 19530)")
    parser.add_argument("--user", default=None, help="Milvus username (if auth enabled)")
    parser.add_argument("--password", default=None, help="Milvus password (if auth enabled)")
    parser.add_argument("--input", required=True, help="Input directory containing exported JSON files")
    parser.add_argument("--batch-size", type=int, default=200, help="Insert batch size (default: 200)")
    parser.add_argument("--drop-existing", action="store_true", help="Drop existing collection before importing")
    args = parser.parse_args()

    print("=" * 60)
    print("Milvus Data Import Tool")
    print("=" * 60)

    # Connect
    connect_milvus(args.host, args.port, args.user, args.password)

    # Load files
    print("\n[1/4] Loading export files...")
    schema_info, index_info, all_data = load_export_files(args.input)

    # Create collection
    print("\n[2/4] Creating collection...")
    collection = create_collection(schema_info, drop_existing=args.drop_existing)

    # Insert data
    print(f"\n[3/4] Importing data ({len(all_data)} records)...")
    import_start = time.time()
    total_inserted = insert_data(collection, schema_info, all_data, batch_size=args.batch_size)
    import_time = time.time() - import_start
    print(f"  Flush completed. Total entities: {collection.num_entities}")

    # Create indexes
    print("\n[4/4] Creating indexes...")
    index_start = time.time()
    create_indexes(collection, index_info)
    index_time = time.time() - index_start

    # Load collection
    collection.load()
    print(f"  Collection loaded and ready for queries")

    # Save import metadata
    import_meta = {
        "collection_name": schema_info["collection_name"],
        "total_records": total_inserted,
        "import_time_seconds": round(import_time, 2),
        "index_time_seconds": round(index_time, 2),
        "target_host": args.host,
        "target_port": args.port,
        "import_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "tool_version": "1.0.0",
    }
    meta_path = os.path.join(args.input, "import_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(import_meta, f, indent=2)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Import completed successfully!")
    print(f"  Records:    {total_inserted}")
    print(f"  Import:     {import_time:.2f}s")
    print(f"  Indexing:   {index_time:.2f}s")
    print(f"  Collection: {schema_info['collection_name']}")
    print(f"{'=' * 60}")

    connections.disconnect("default")


if __name__ == "__main__":
    main()
