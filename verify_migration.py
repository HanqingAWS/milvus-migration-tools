#!/usr/bin/env python3
"""
Milvus Migration Verification Tool
====================================
Compare data between source and target Milvus instances after migration.

Usage:
    python3 verify_migration.py \
        --source-host 10.0.1.100 --source-port 19530 \
        --target-host 10.0.2.200 --target-port 19530 \
        --collection game_cs_knowledge

Checks performed:
    1. Collection existence on both instances
    2. Record count consistency
    3. Schema field matching
    4. Sample data comparison (first 10 records)
    5. Random spot check (20 records)
    6. Vector search result consistency (Top-10)
"""

import argparse
import json
import random
import sys
import time

import numpy as np
from pymilvus import Collection, connections, utility


class Verifier:
    def __init__(self):
        self.results = {"checks": [], "passed": 0, "failed": 0}

    def check(self, name, condition, detail=""):
        status = "PASS" if condition else "FAIL"
        self.results["checks"].append(
            {"name": name, "status": status, "detail": detail}
        )
        if condition:
            self.results["passed"] += 1
            print(f"  [PASS] {name}")
            if detail:
                print(f"         {detail}")
        else:
            self.results["failed"] += 1
            print(f"  [FAIL] {name}")
            if detail:
                print(f"         {detail}")

    @property
    def all_passed(self):
        return self.results["failed"] == 0


def connect_both(source_host, source_port, target_host, target_port, user=None, password=None):
    """Connect to both source and target Milvus instances."""
    kwargs_s = {"host": source_host, "port": source_port}
    kwargs_t = {"host": target_host, "port": target_port}
    if user and password:
        kwargs_s["user"] = user
        kwargs_s["password"] = password
        kwargs_t["user"] = user
        kwargs_t["password"] = password

    connections.connect("source", **kwargs_s)
    print(f"Connected to SOURCE: {source_host}:{source_port}")
    connections.connect("target", **kwargs_t)
    print(f"Connected to TARGET: {target_host}:{target_port}")


def find_primary_field(collection):
    """Find the primary key field name."""
    for f in collection.schema.fields:
        if f.is_primary:
            return f.name
    return "id"


def find_vector_field(collection):
    """Find the first vector field name."""
    for f in collection.schema.fields:
        if "VECTOR" in str(f.dtype):
            return f.name
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Verify Milvus migration data consistency"
    )
    parser.add_argument("--source-host", required=True, help="Source Milvus host")
    parser.add_argument("--source-port", default="19530", help="Source Milvus port")
    parser.add_argument("--target-host", required=True, help="Target Milvus host")
    parser.add_argument("--target-port", default="19530", help="Target Milvus port")
    parser.add_argument("--user", default=None, help="Milvus username")
    parser.add_argument("--password", default=None, help="Milvus password")
    parser.add_argument("--collection", required=True, help="Collection name to verify")
    parser.add_argument("--output", default=None, help="Output JSON file for results")
    parser.add_argument("--spot-check-count", type=int, default=20, help="Number of random records to spot check (default: 20)")
    args = parser.parse_args()

    print("=" * 60)
    print("Milvus Migration Verification Tool")
    print("=" * 60)

    # Connect
    connect_both(
        args.source_host, args.source_port,
        args.target_host, args.target_port,
        args.user, args.password,
    )

    verifier = Verifier()
    collection_name = args.collection

    # ===== Check 1: Collection exists =====
    print("\n--- Collection Existence ---")
    has_s = utility.has_collection(collection_name, using="source")
    has_t = utility.has_collection(collection_name, using="target")
    verifier.check("Collection exists on SOURCE", has_s, f"'{collection_name}': {has_s}")
    verifier.check("Collection exists on TARGET", has_t, f"'{collection_name}': {has_t}")

    if not (has_s and has_t):
        print("\nCannot continue: collection missing on one or both instances.")
        sys.exit(1)

    col_s = Collection(collection_name, using="source")
    col_t = Collection(collection_name, using="target")
    col_s.load()
    col_t.load()

    pk_field = find_primary_field(col_s)
    vec_field = find_vector_field(col_s)

    # ===== Check 2: Record count =====
    print("\n--- Record Count ---")
    count_s = col_s.num_entities
    count_t = col_t.num_entities
    verifier.check("Record count matches", count_s == count_t, f"Source: {count_s}, Target: {count_t}")

    # ===== Check 3: Schema =====
    print("\n--- Schema Comparison ---")
    schema_match = True
    for fs, ft in zip(col_s.schema.fields, col_t.schema.fields):
        if fs.name != ft.name or str(fs.dtype) != str(ft.dtype):
            schema_match = False
            break
    field_names = [f.name for f in col_s.schema.fields]
    verifier.check("Schema matches", schema_match, f"Fields: {field_names}")

    # ===== Check 4: Sample data (first 10) =====
    print("\n--- Sample Data Comparison (first 10 records) ---")
    output_fields = [f.name for f in col_s.schema.fields]
    scalar_fields = [f.name for f in col_s.schema.fields if "VECTOR" not in str(f.dtype)]

    sample_s = col_s.query(expr=f"{pk_field} >= 0", output_fields=output_fields, limit=10)
    sample_t = col_t.query(expr=f"{pk_field} >= 0", output_fields=output_fields, limit=10)
    sample_s.sort(key=lambda x: x[pk_field])
    sample_t.sort(key=lambda x: x[pk_field])

    sample_match = True
    sample_detail = "All scalar fields and vectors match"
    for rs, rt in zip(sample_s, sample_t):
        for field in scalar_fields:
            if rs.get(field) != rt.get(field):
                sample_match = False
                sample_detail = f"Mismatch at {pk_field}={rs[pk_field]} field='{field}'"
                break
        if vec_field and sample_match:
            vec_s = np.array(rs[vec_field])
            vec_t = np.array(rt[vec_field])
            cos_sim = np.dot(vec_s, vec_t) / (np.linalg.norm(vec_s) * np.linalg.norm(vec_t))
            if cos_sim < 0.9999:
                sample_match = False
                sample_detail = f"Vector mismatch at {pk_field}={rs[pk_field]}, cosine_sim={cos_sim:.6f}"
        if not sample_match:
            break
    verifier.check("Sample data matches", sample_match, sample_detail)

    # ===== Check 5: Random spot check =====
    print("\n--- Random Spot Check ---")
    random.seed(42)
    spot_count = min(args.spot_check_count, count_s)
    spot_ids = random.sample(range(1, count_s + 1), spot_count)
    spot_expr = f"{pk_field} in {spot_ids}"

    spot_s = col_s.query(expr=spot_expr, output_fields=output_fields)
    spot_t = col_t.query(expr=spot_expr, output_fields=output_fields)
    spot_s.sort(key=lambda x: x[pk_field])
    spot_t.sort(key=lambda x: x[pk_field])

    spot_match = len(spot_s) == len(spot_t)
    spot_detail = f"Checked {spot_count} random records"
    if spot_match:
        for rs, rt in zip(spot_s, spot_t):
            for field in scalar_fields:
                if rs.get(field) != rt.get(field):
                    spot_match = False
                    spot_detail = f"Mismatch at {pk_field}={rs[pk_field]} field='{field}'"
                    break
            if vec_field and spot_match:
                vec_s = np.array(rs[vec_field])
                vec_t = np.array(rt[vec_field])
                cos_sim = np.dot(vec_s, vec_t) / (np.linalg.norm(vec_s) * np.linalg.norm(vec_t))
                if cos_sim < 0.9999:
                    spot_match = False
                    spot_detail = f"Vector mismatch at {pk_field}={rs[pk_field]}, cosine_sim={cos_sim:.6f}"
            if not spot_match:
                break
    verifier.check(f"Random spot check ({spot_count} records)", spot_match, spot_detail)

    # ===== Check 6: Search consistency =====
    if vec_field:
        print("\n--- Search Result Consistency (Top-10) ---")
        test_record = col_s.query(expr=f"{pk_field} >= 0", output_fields=[vec_field], limit=1)
        if test_record:
            test_vec = test_record[0][vec_field]

            # Detect metric type from index
            metric_type = "COSINE"
            for idx in col_s.indexes:
                if idx.field_name == vec_field:
                    metric_type = idx.params.get("metric_type", "COSINE")
                    break

            search_params = {"metric_type": metric_type, "params": {"nprobe": 10}}
            res_s = col_s.search(data=[test_vec], anns_field=vec_field, param=search_params, limit=10)
            res_t = col_t.search(data=[test_vec], anns_field=vec_field, param=search_params, limit=10)

            search_match = True
            search_details = []
            for i, (hs, ht) in enumerate(zip(res_s[0], res_t[0])):
                if hs.id != ht.id:
                    search_match = False
                    search_details.append(f"Rank {i+1}: source.id={hs.id} vs target.id={ht.id}")
                score_diff = abs(hs.score - ht.score)
                if score_diff > 0.001:
                    search_match = False
                    search_details.append(f"Rank {i+1}: score_diff={score_diff:.6f}")

            detail = "; ".join(search_details[:3]) if search_details else "Top-10 results identical"
            verifier.check("Search results consistent", search_match, detail)

    # ===== Summary =====
    total_checks = len(verifier.results["checks"])
    print(f"\n{'=' * 60}")
    print(f"Verification: {verifier.results['passed']} PASSED, {verifier.results['failed']} FAILED out of {total_checks} checks")
    print(f"{'=' * 60}")

    # Save results
    if args.output:
        verifier.results["source"] = f"{args.source_host}:{args.source_port}"
        verifier.results["target"] = f"{args.target_host}:{args.target_port}"
        verifier.results["collection"] = collection_name
        verifier.results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(verifier.results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")

    connections.disconnect("source")
    connections.disconnect("target")

    sys.exit(0 if verifier.all_passed else 1)


if __name__ == "__main__":
    main()
