from collections import defaultdict
import sys
import time
from pathlib import Path

#Actual algorithm starts here

# Starting from mandatory sets, go through the list of subsets and find one that fills in everything.
# If that can't be found, use the first one that adds something new.
# Repeat until a set cover is made
# Remove the last subset added and retry with another one.

def isSolvedHelper(solution, universal):
    remaining = set(universal)
    for list in solution:
        for elem in list:
            try:
                remaining.remove(elem)
            except:
                continue
    if len(remaining) == 0:
        return True
    return False

# checks if added subset contains a new element.
def addsNewHelper(rem_elems, s):
    for elem in s:
        if elem in rem_elems:
            return True
    return False
    
def updateRemElems(rem_elems, curr_solution):
    for list in curr_solution:
        for elem in list:
            if elem in rem_elems:
                rem_elems.remove(elem)
    return rem_elems
def setCover(rem_sets, rem_elems, universal, mandatory_sets):
    min_solution = []
    curr_solution = mandatory_sets.copy()   
    def backtrack(rem_sets, curr_solution, rem_elems, universal):
        nonlocal min_solution
        # Base case: when rem_elems is zero.
        if len(rem_elems) < 1:
            if (len(min_solution) == 0 or len(curr_solution) < len(min_solution)):
                min_solution = curr_solution[:]
            return
        # Second optimization: if curr_solution is worse, just stop man
        if len(min_solution) > 0 and len(curr_solution) >= len(min_solution):
            return
        # Go through the remaining sets.
        for i in range(len(rem_sets)):
            s = rem_sets[i]
            
            # Skip sets that don't add new elements
            if addsNewHelper(rem_elems, s) == False:
                continue  # ✅ FIXED: Just skip, don't mutate
            
            # Choose this set
            curr_solution.append(s)
            
            # Check if adding this set solves it
            if isSolvedHelper(curr_solution, universal):
                if len(min_solution) == 0 or len(curr_solution) < len(min_solution):
                    min_solution = curr_solution[:]
                curr_solution.pop()  # ✅ FIXED: Backtrack
                continue  # ✅ FIXED: Keep exploring, don't return
            
            # Create new state for recursion (don't mutate shared state)
            new_rem_elems = rem_elems.copy()
            new_rem_elems = updateRemElems(new_rem_elems, curr_solution)
            new_rem_sets = rem_sets[:i] + rem_sets[i+1:]
            
            # Recurse
            backtrack(new_rem_sets, curr_solution, new_rem_elems, universal)
            
            # Backtrack
            curr_solution.pop()  # ✅ FIXED: Always backtrack
            
    backtrack(rem_sets, curr_solution, rem_elems, universal)
    return min_solution

def verifyCover(solution, universal):
    # remove each index from the solution.
    covered = set()
    for subset in solution:
        covered.update(subset)
    return covered == set(universal)


def main(filepath):
    # First parse the given file.
    content = ""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        print("Failed to parse using utf-8")

    lines = content.split('\n')
    U_size = int(lines[0].strip()) # converting from str into int as even though we get a number it's still a string.
    S_size = int(lines[1].strip())
    SUBSETS = []
    UNIVERSAL_SET = list(range(1, U_size+1))

    for line in lines[2:-1]: #[2:-1] meaning start from index 2 until the second last element
        line_arr = line.split()
        for i,x in enumerate(line_arr):
            line_arr[i] = int(x)
        SUBSETS.append(line_arr)

    # print(S_size)
    # print(len(SUBSETS))
    # print(SUBSETS)
    assert len(SUBSETS) == S_size, "Given S size and actual is not equal. Likely parse error."

    indices = defaultdict(list) # default dict creates a set when we need a default value.
    # First optimization is to create a list of sets that MUST be included.
    for i,subset in enumerate(SUBSETS): # gives us both index and the element
        for elem in subset: # for each element in the Subset
            indices[elem].append(i)

    mandatory_sets = []
    rem_elems = set(UNIVERSAL_SET) # converts to set and deep clones at the same time
    for u in UNIVERSAL_SET: # create mandatory sets and removes the elements it covers.
        if len(indices[u]) == 1:
            mandatory_sets.append(SUBSETS[indices[u][0]])
            rem_elems.remove(u)

    rem_sets = []
    for s in SUBSETS:
        if s not in mandatory_sets:
            rem_sets.append(set(s))
    #print(rem_sets)
    #min_set_cover_len = len(setCover(rem_sets, rem_elems, UNIVERSAL_SET, mandatory_sets))
    start = time.perf_counter()
    set_cover = setCover(rem_sets, rem_elems, UNIVERSAL_SET, mandatory_sets)
    end = time.perf_counter()
    print("Minimum Set Cover Length: " + str(len(set_cover)))
    print(set_cover)
    print("Time elapsed: " + str(end-start))
    print("Verifying cover: " + str(verifyCover(set_cover, UNIVERSAL_SET)))
    #print(str(UNIVERSAL_SET))
    return [len(set_cover), setCover(rem_sets, rem_elems, UNIVERSAL_SET, mandatory_sets)]
test_dir = Path(r"C:/Users/heinz/Desktop/CS_Projects/cse373/test_data")
main(test_dir)