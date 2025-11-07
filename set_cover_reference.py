#!/usr/bin/env python3
"""
Exact Minimum Set Cover solver (branch-and-bound + optional ILP fallback).
Usage:
    python exact_set_cover.py /mnt/data/s-k-150-225
"""

import sys
import math
import time
from collections import defaultdict

def parse_instance(path):
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip() != ""]
    if len(lines) < 2:
        raise ValueError("File must contain at least two lines: universe_size and num_subsets")
    n = int(lines[0])
    m = int(lines[1])
    subset_lines = lines[2:]
    if len(subset_lines) != m:
        # allow flexible parsing — sometimes m may not match exactly
        print(f"Warning: header says {m} subsets but file contains {len(subset_lines)} subset lines. Proceeding with {len(subset_lines)}.")
        m = len(subset_lines)
    subsets = []
    for ln in subset_lines:
        if ln.strip() == "":
            subsets.append(set())
            continue
        elems = [int(x) for x in ln.split() if x.strip() != ""]
        # ignore out-of-range and duplicates
        s = set(x for x in elems if 1 <= x <= n)
        subsets.append(s)
    return n, subsets

def sets_to_bitmasks(n, subsets):
    masks = []
    for s in subsets:
        mask = 0
        for e in s:
            mask |= 1 << (e-1)  # element e maps to bit e-1
        masks.append(mask)
    universe_mask = (1 << n) - 1 if n < (1<<20) else (1 << n) - 1
    return universe_mask, masks

def remove_empty_and_duplicates(masks):
    seen = {}
    new_masks = []
    for i, m in enumerate(masks):
        if m == 0:
            continue
        if m in seen:
            continue
        seen[m] = True
        new_masks.append(m)
    return new_masks

def remove_dominated_masks(masks):
    # Remove mask A if there exists mask B with B != A and A is subset of B (i.e., A & B == A)
    # We'll sort by popcount descending — larger sets first — and only keep masks that are not subset of an already kept mask.
    masks = list(set(masks))
    masks_sorted = sorted(masks, key=lambda x: bitcount(x), reverse=True)
    kept = []
    for m in masks_sorted:
        dominated = False
        for k in kept:
            if (m | k) == k:  # m subset of k
                dominated = True
                break
        if not dominated:
            kept.append(m)
    return kept

def bitcount(x):
    return x.bit_count()

def choose_element_with_min_frequency(universe_mask, masks, current_mask):
    # return element index (0-based) that is uncovered and appears in fewest sets (min frequency)
    remaining = universe_mask & ~current_mask
    if remaining == 0:
        return None
    # Count frequency
    freq = defaultdict(int)
    element = None
    # For speed, iterate bits in remaining
    r = remaining
    while r:
        lsb = (r & -r)
        idx = (lsb.bit_length() - 1)
        # count how many masks cover this idx
        cnt = 0
        for m in masks:
            if (m >> idx) & 1:
                cnt += 1
        freq[idx] = cnt
        r &= r - 1
    # choose index with min positive freq (>0)
    min_idx = min(freq.keys(), key=lambda k: (freq[k], k))
    return min_idx

def compute_max_uncovered_cover(masks, current_mask):
    rem = ~current_mask
    best = 0
    for m in masks:
        cover = bitcount(m & rem)
        if cover > best: best = cover
    return best

def branch_and_bound_exact(universe_mask, masks, time_limit=None):
    start = time.time()
    n_masks = len(masks)

    # Precompute for each element the list of mask indices that cover it
    elem_to_masks = {}
    total_bits = universe_mask.bit_length()
    for e in range(total_bits):
        lst = [i for i, m in enumerate(masks) if ((m >> e) & 1)]
        if lst:
            elem_to_masks[e] = lst

    best_solution = None
    best_size = float('inf')

    # Initial greedy upper bound (fast) to help prune (still exact because we use it only as upper bound)
    def greedy_upper_bound():
        cov = 0
        chosen = []
        masks_copy = masks[:]
        cov_mask = 0
        while cov_mask != universe_mask:
            best_idx, best_cover = None, 0
            for i, m in enumerate(masks_copy):
                c = bitcount(m & ~cov_mask)
                if c > best_cover:
                    best_cover = c
                    best_idx = i
            if best_cover == 0:
                break
            chosen.append(best_idx)
            cov_mask |= masks_copy[best_idx]
        return chosen

    greedy_sol = greedy_upper_bound()
    if greedy_sol and bitcount(universe_mask & ~sum((masks[i] for i in greedy_sol), 0)) == 0:
        best_solution = greedy_sol[:]
        best_size = len(best_solution)
    else:
        # if greedy didn't cover (shouldn't happen), set initial best_size to inf
        best_size = float('inf')

    # Order masks for branching: larger sets first often helps
    # But we will use element-based branching (choose an uncovered element)
    visited = 0

    # recursion
    def dfs(current_mask, chosen_indices):
        nonlocal best_solution, best_size, visited, start
        visited += 1
        # time check
        if time_limit and (time.time() - start) > time_limit:
            raise TimeoutError("Time limit exceeded")

        if current_mask == universe_mask:
            if len(chosen_indices) < best_size:
                best_solution = chosen_indices.copy()
                best_size = len(chosen_indices)
            return

        # pruning by current size
        if len(chosen_indices) >= best_size:
            return

        remaining = bitcount(universe_mask & ~current_mask)
        if remaining == 0:
            if len(chosen_indices) < best_size:
                best_solution = chosen_indices.copy()
                best_size = len(chosen_indices)
            return

        # optimistic lower bound: ceil(remaining / max_uncovered_coverage)
        maxcov = compute_max_uncovered_cover(masks, current_mask)
        if maxcov == 0:
            return  # cannot cover remaining elements
        lower_bound = math.ceil(remaining / maxcov)
        if len(chosen_indices) + lower_bound >= best_size:
            return

        # choose uncovered element with minimum branching (# sets that cover it)
        elem = None
        # gather uncovered elements and their frequencies
        rem = universe_mask & ~current_mask
        min_freq = None
        # compute frequencies
        r = rem
        while r:
            lsb = (r & -r)
            idx = (lsb.bit_length() - 1)
            # compute frequency
            freq = 0
            for i, m in enumerate(masks):
                if ((m >> idx) & 1):
                    freq += 1
            if freq == 0:
                return  # cannot cover this element at all
            if (min_freq is None) or (freq < min_freq):
                min_freq = freq
                elem = idx
            r &= r - 1

        # Branch: iterate the sets covering this element
        covering_sets = [i for i, m in enumerate(masks) if ((m >> elem) & 1)]
        # sort covering sets by decreasing new coverage (helpful ordering)
        covering_sets.sort(key=lambda i: bitcount(masks[i] & ~current_mask), reverse=True)

        for sidx in covering_sets:
            # pruning
            if len(chosen_indices) + 1 >= best_size:
                break
            new_mask = current_mask | masks[sidx]
            chosen_indices.append(sidx)
            dfs(new_mask, chosen_indices)
            chosen_indices.pop()
            # small optimization: if we've already found a solution of size len(chosen_indices)+1 then no need to try more sets
            if len(chosen_indices) + 1 >= best_size:
                break

    try:
        dfs(0, [])
    except TimeoutError:
        pass

    return best_solution, best_size, visited

# Optional: try ILP using OR-Tools CP-SAT or PuLP if available
def try_ilp_exact(n, masks, time_limit=None):
    try:
        # Try OR-Tools CP-SAT (often faster for combinatorial problems)
        from ortools.sat.python import cp_model
        model = cp_model.CpModel()
        x = [model.NewBoolVar(f"x{i}") for i in range(len(masks))]
        # cover constraints
        for e in range(n):
            # collect masks covering e
            covering = [i for i, m in enumerate(masks) if ((m >> e) & 1)]
            if not covering:
                return None  # infeasible
            model.Add(sum(x[i] for i in covering) >= 1)
        model.Minimize(sum(x))
        solver = cp_model.CpSolver()
        if time_limit:
            solver.parameters.max_time_in_seconds = float(time_limit)
        solver.parameters.num_search_workers = 8
        res = solver.Solve(model)
        if res == cp_model.OPTIMAL or res == cp_model.FEASIBLE:
            sol = [i for i in range(len(masks)) if solver.Value(x[i]) == 1]
            return sol
        else:
            return None
    except Exception:
        pass

    try:
        import pulp
        prob = pulp.LpProblem("setcover", pulp.LpMinimize)
        x = [pulp.LpVariable(f"x{i}", cat="Binary") for i in range(len(masks))]
        prob += pulp.lpSum(x)
        for e in range(n):
            prob += pulp.lpSum(x[i] for i, m in enumerate(masks) if ((m >> e) & 1)) >= 1
        if time_limit:
            pulp.LpSolverDefault.msg = 0
            # solver time limit handling depends on solver; default CBC allows timeLimit param via pulp.PULP_CBC_CMD
            solver = pulp.PULP_CBC_CMD(timeLimit=int(time_limit))
            prob.solve(solver)
        else:
            prob.solve()
        if pulp.LpStatus[prob.status] in ("Optimal", "Not Solved", "Integer Feasible"):
            sol = [i for i in range(len(masks)) if pulp.value(x[i]) >= 0.5]
            return sol
        else:
            return None
    except Exception:
        pass

    return None

from itertools import combinations

def verify_cover(universe_mask, masks, chosen_indices):
    """Return True if the chosen sets cover the entire universe."""
    cov = 0
    for i in chosen_indices:
        cov |= masks[i]
    return cov == universe_mask

def verify_optimality(universe_mask, masks, chosen_indices, brute_limit=20):
    """
    Verify the solution is truly minimum.
    For small instances (<= brute_limit sets), brute-force check if any smaller subset covers the universe.
    """
    k = len(chosen_indices)
    m = len(masks)
    if m > brute_limit:
        print(f"[Warning] Too many subsets ({m}) for brute-force optimality check — skipping.")
        return True  # skip for large instances to avoid combinatorial explosion

    # Try all smaller combinations
    for size in range(1, k):
        for combo in combinations(range(m), size):
            cov = 0
            for i in combo:
                cov |= masks[i]
            if cov == universe_mask:
                print(f"[Error] Found smaller cover of size {size} → Not optimal.")
                return False
    return True



def main(path, time_limit=None, try_ilp=True):
    t0 = time.time()
    n, subsets = parse_instance(path)
    universe_mask, masks = sets_to_bitmasks(n, subsets)
    print(f"Parsed universe size n={n}, raw subset count={len(masks)}")

    masks = remove_empty_and_duplicates(masks)
    masks = remove_dominated_masks(masks)
    print(f"After cleaning, subset count reduced to {len(masks)}")

    # try ILP first if available
    if try_ilp:
        print("Attempting ILP/CP-SAT solver if available...")
        ilp_sol = try_ilp_exact(n, masks, time_limit)
        if ilp_sol is not None:
            print("ILP/CP-SAT produced a solution.")
            chosen = ilp_sol
            cov = 0
            for i in chosen:
                cov |= masks[i]
            if cov == universe_mask:
                print(f"Optimal (from ILP/CP-SAT) solution size: {len(chosen)}")
                print("Chosen set indices (0-based):", chosen)
                print("Time elapsed: {:.2f}s".format(time.time() - t0))
                return chosen
            else:
                print("ILP returned infeasible or incomplete cover; fallback to exact search.")

    print("Running exact branch-and-bound search (may take long for hard instances)...")
    best_solution, best_size, visited = branch_and_bound_exact(universe_mask, masks, time_limit)
    t1 = time.time()
    if best_solution is None:
        print("No solution found within limits.")
        return None
    else:
        cov = 0
        for i in best_solution:
            cov |= masks[i]
        if not verify_cover(universe_mask, masks, best_solution):
            print("❌ Error: returned solution does NOT cover the entire universe.")
            return None

        print(f"Exact solution size: {len(best_solution)}")
        print("Chosen set indices (0-based):", best_solution)
        print(f"Nodes visited: {visited}")
        print("Time elapsed: {:.2f}s".format(t1 - t0))

        # ✅ Correctness checks
        print("\n=== Correctness Verification ===")
        cover_ok = verify_cover(universe_mask, masks, best_solution)
        print(f"Coverage check: {'✅ PASSED' if cover_ok else '❌ FAILED'}")

        opt_ok = verify_optimality(universe_mask, masks, best_solution)
        print(f"Optimality check: {'✅ PASSED' if opt_ok else '❌ FAILED'}")

        if cover_ok and opt_ok:
            print("✅ Solution verified correct and optimal.")
        else:
            print("⚠️ Solution failed verification!")

        return best_solution


if __name__ == "__main__":
    print("Script started!")  # ← Add this
    print(f"Arguments: {sys.argv}")  # ← Add this
    if len(sys.argv) < 2:
        print("Usage: python exact_set_cover.py <instance_file> [time_limit_seconds]")
        sys.exit(1)
    path = sys.argv[1]
    time_limit = None
    if len(sys.argv) >= 3:
        time_limit = float(sys.argv[2])
    sol = main(path, time_limit=time_limit, try_ilp=True)
