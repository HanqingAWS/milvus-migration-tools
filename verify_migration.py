#!/usr/bin/env python3
"""
Milvus Migration Verification Tool
====================================
Compare data between source and target Milvus instances after migration.
Automatically detects primary key, vector fields, and data types.
No assumptions about data structure - works with any collection schema.

Usage:
    python3 verify_migration.py \
        --source-host 10.0.1.100 --source-port 19530 \
        --target-host 10.0.2.200 --target-port 19530 \
        --collection my_collection

Checks performed:
    1. Collection existence on both instances
    2. Record count consistency
    3. Schema field matching
    4. Sample data comparison (10 records)
    5. Random spot check (20 records by PK)
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
    """Auto-detect primary key field name and dtype."""
    for f in collection.schema.fields:
        if f.is_primary:
            return f.name, str(f.dtype)
    return None, None


def find_vector_field(collection):
    """Find the first vector field name."""
    for f in collection.schema.fields:
        if "VECTOR" in str(f.dtype):
            return f.name
    return None


def get_scalar_fields(collection):
    """Get all non-vector field names."""
    return [f.name for f in collection.schema.fields if "VECTOR" not in str(f.dtype)]


def build_match_all_expr(pk_name, pk_dtype):
    """Build an expression that matches all records, regardless of PK type."""
    if "INT" in pk_dtype:
        return f"{pk_name} >= -9223372036854775808"
    else:
        # VARCHAR PK
        return f'{pk_name} != "__IMPOSSIBLE_VALUE_PLACEHOLDER__"'


def build_pk_in_expr(pk_name, pk_dtype, pk_values):
    """Build a 'pk IN [...]' expression that works for both INT and VARCHAR PKs."""
    if "INT" in pk_dtype:
        return f"{pk_name} in {pk_values}"
    else:
        # VARCHAR PK - values need to be quoted
        quoted = [f'"{v}"' for v in pk_values]
        return f"{pk_name} in [{', '.join(quoted)}]"


def compare_rows(row_s, row_t, scalar_fields, vec_field, pk_name):
    """Compare two rows field by field. Returns (match: bool, detail: str)."""
    pk_val = row_s.get(pk_name, "?")

    # Compare scalar fields
    for field in scalar_fields:
        val_s = row_s.get(field)
        val_t = row_t.get(field)
        if val_s != val_t:
            return False, f"Mismatch at PK={pk_val} field='{field}': '{val_s}' vs '{val_t}'"

    # Compare vector field
    if vec_field and vec_field in row_s and vec_field in row_t:
        vec_s = np.array(row_s[vec_field], dtype=np.float32)
        vec_t = np.array(row_t[vec_field], dtype=np.float32)
        norm_s = np.linalg.norm(vec_s)
        norm_t = np.linalg.norm(vec_t)
        if norm_s > 0 and norm_t > 0:
            cos_sim = np.dot(vec_s, vec_t) / (norm_s * norm_t)
            if cos_sim < 0.9999:
                return False, f"Vector mismatch at PK={pk_val}, cosine_sim={cos_sim:.6f}"

    return True, ""


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

    # Auto-detect fields
    pk_name, pk_dtype = find_primary_field(col_s)
    vec_field = find_vector_field(col_s)
    scalar_fields = get_scalar_fields(col_s)
    output_fields = [f.name for f in col_s.schema.fields]
    match_all_expr = build_match_all_expr(pk_name, pk_dtype)

    print(f"\n  Auto-detected: PK='{pk_name}' ({pk_dtype}), Vector='{vec_field}'")

    # ===== Check 2: Record count =====
    print("\n--- Record Count ---")
    count_s = col_s.num_entities
    count_t = col_t.num_entities
    verifier.check("Record count matches", count_s == count_t, f"Source: {count_s}, Target: {count_t}")

    # ===== Check 3: Schema =====
    print("\n--- Schema Comparison ---")
    schema_match = True
    schema_mismatch_detail = ""
    if len(col_s.schema.fields) != len(col_t.schema.fields):
        schema_match = False
        schema_mismatch_detail = f"Field count: source={len(col_s.schema.fields)} vs target={len(col_t.schema.fields)}"
    else:
        for fs, ft in zip(col_s.schema.fields, col_t.schema.fields):
            if fs.name != ft.name or str(fs.dtype) != str(ft.dtype):
                schema_match = False
                schema_mismatch_detail = f"'{fs.name}'({fs.dtype}) vs '{ft.name}'({ft.dtype})"
                break
    field_names = [f.name for f in col_s.schema.fields]
    verifier.check(
        "Schema matches", schema_match,
        schema_mismatch_detail or f"Fields: {field_names}"
    )

    # ===== Check 4: Sample data (first 10) =====
    print("\n--- Sample Data Comparison (10 records) ---")
    sample_s = col_s.query(expr=match_all_expr, output_fields=output_fields, limit=10)
    sample_t = col_t.query(expr=match_all_expr, output_fields=output_fields, limit=10)
    sample_s.sort(key=lambda x: str(x[pk_name]))
    sample_t.sort(key=lambda x: str(x[pk_name]))

    sample_match = True
    sample_detail = "All scalar fields and vectors match"
    if len(sample_s) != len(sample_t):
        sample_match = False
        sample_detail = f"Sample size mismatch: source={len(sample_s)} vs target={len(sample_t)}"
    else:
        for rs, rt in zip(sample_s, sample_t):
            match, detail = compare_rows(rs, rt, scalar_fields, vec_field, pk_name)
            if not match:
                sample_match = False
                sample_detail = detail
                break
    verifier.check("Sample data matches", sample_match, sample_detail)

    # ===== Check 5: Random spot check =====
    print("\n--- Random Spot Check ---")
    # First, fetch a batch of PKs from source to pick random ones from
    all_pks_s = col_s.query(
        expr=match_all_expr,
        output_fields=[pk_name],
        limit=min(count_s, 16384),  # Get up to 16K PKs for random selection
    )
    pk_values = [row[pk_name] for row in all_pks_s]

    spot_count = min(args.spot_check_count, len(pk_values))
    random.seed(42)
    sampled_pks = random.sample(pk_values, spot_count)

    spot_expr = build_pk_in_expr(pk_name, pk_dtype, sampled_pks)
    spot_s = col_s.query(expr=spot_expr, output_fields=output_fields)
    spot_t = col_t.query(expr=spot_expr, output_fields=output_fields)

    # Build lookup dicts by PK for matching
    dict_s = {str(row[pk_name]): row for row in spot_s}
    dict_t = {str(row[pk_name]): row for row in spot_t}

    spot_match = True
    spot_detail = f"Checked {spot_count} random records"

    if len(spot_s) != len(spot_t):
        spot_match = False
        spot_detail = f"Record count mismatch: source={len(spot_s)} vs target={len(spot_t)}"
    else:
        for pk_str, rs in dict_s.items():
            rt = dict_t.get(pk_str)
            if rt is None:
                spot_match = False
                spot_detail = f"PK={pk_str} exists in source but missing in target"
                break
            match, detail = compare_rows(rs, rt, scalar_fields, vec_field, pk_name)
            if not match:
                spot_match = False
                spot_detail = detail
                break

    verifier.check(f"Random spot check ({spot_count} records)", spot_match, spot_detail)

    # ===== Check 6: Search consistency =====
    if vec_field:
        print("\n--- Search Result Consistency (Top-10) ---")
        test_record = col_s.query(expr=match_all_expr, output_fields=[vec_field], limit=1)
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
                    search_details.append(f"Rank {i+1}: source={hs.id} vs target={ht.id}")
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
        verifier.results["primary_key"] = f"{pk_name} ({pk_dtype})"
        verifier.results["vector_field"] = vec_field
        verifier.results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(verifier.results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")

    connections.disconnect("source")
    connections.disconnect("target")

    sys.exit(0 if verifier.all_passed else 1)


if __name__ == "__main__":
    main()
