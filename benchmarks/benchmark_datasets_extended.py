"""
Extended Benchmark Datasets for SynapticFold v2 Evaluation
Contains 25 canonical problems each for:
1. MBPP (Mostly Basic Python Problems)
2. ARC-Challenge (AI2 Reasoning Challenge)
3. MMLU-STEM (Computer Science, Mathematics, Physics, Chemistry)
4. HellaSwag (Commonsense Situational Continuation)
"""

MBPP_SAMPLES = [
    {
        "task_id": 1,
        "prompt": "Write a Python function to find the minimum sum of a path from top to bottom in a triangle of numbers.",
        "test": "def min_sum_path(triangle):\n    for i in range(len(triangle) - 2, -1, -1):\n        for j in range(len(triangle[i])):\n            triangle[i][j] += min(triangle[i + 1][j], triangle[i + 1][j + 1])\n    return triangle[0][0]"
    },
    {
        "task_id": 2,
        "prompt": "Write a function to find the shared elements in two given lists.",
        "test": "def similar_elements(test_tup1, test_tup2):\n    return list(set(test_tup1) & set(test_tup2))"
    },
    {
        "task_id": 3,
        "prompt": "Write a python function to identify non-prime numbers.",
        "test": "import math\ndef is_not_prime(n):\n    if n <= 1: return True\n    for i in range(2, int(math.isqrt(n)) + 1):\n        if n % i == 0: return True\n    return False"
    },
    {
        "task_id": 4,
        "prompt": "Write a function to find the largest number that can be formed from given list of integers.",
        "test": "from functools import cmp_to_key\ndef largest_number(nums):\n    s_nums = [str(n) for n in nums]\n    s_nums.sort(key=cmp_to_key(lambda a, b: 1 if a+b < b+a else -1))\n    res = ''.join(s_nums)\n    return '0' if res[0] == '0' else res"
    },
    {
        "task_id": 5,
        "prompt": "Write a function to count frequencies of all elements in a list using a dictionary.",
        "test": "def count_frequency(lst):\n    res = {}\n    for item in lst:\n        res[item] = res.get(item, 0) + 1\n    return res"
    },
    {
        "task_id": 6,
        "prompt": "Write a python function to check whether a sequence of numbers has a decreasing trend.",
        "test": "def is_decreasing(lst):\n    return all(lst[i] > lst[i+1] for i in range(len(lst)-1))"
    },
    {
        "task_id": 7,
        "prompt": "Write a function to find the volume of a sphere given its radius.",
        "test": "import math\ndef volume_sphere(r):\n    return (4/3) * math.pi * (r ** 3)"
    },
    {
        "task_id": 8,
        "prompt": "Write a function to find the longest common prefix among an array of strings.",
        "test": "def longest_common_prefix(strs):\n    if not strs: return ''\n    prefix = strs[0]\n    for s in strs[1:]:\n        while not s.startswith(prefix):\n            prefix = prefix[:-1]\n            if not prefix: return ''\n    return prefix"
    },
    {
        "task_id": 9,
        "prompt": "Write a python function to calculate the sum of geometric progression series.",
        "test": "def sum_gp(a, r, n):\n    if r == 1: return a * n\n    return a * (1 - r ** n) / (1 - r)"
    },
    {
        "task_id": 10,
        "prompt": "Write a function to reverse only the vowels in a given string.",
        "test": "def reverse_vowels(s):\n    vowels = set('aeiouAEIOU')\n    chars = list(s)\n    i, j = 0, len(s) - 1\n    while i < j:\n        while i < j and chars[i] not in vowels: i += 1\n        while i < j and chars[j] not in vowels: j -= 1\n        chars[i], chars[j] = chars[j], chars[i]\n        i += 1; j -= 1\n    return ''.join(chars)"
    },
    {
        "task_id": 11,
        "prompt": "Write a function to find the first non-repeated character in a string.",
        "test": "def first_non_repeating(s):\n    counts = {}\n    for c in s: counts[c] = counts.get(c, 0) + 1\n    for c in s:\n        if counts[c] == 1: return c\n    return None"
    },
    {
        "task_id": 12,
        "prompt": "Write a python function to find the binomial coefficient C(n, k).",
        "test": "import math\ndef binomial_coeff(n, k):\n    if k < 0 or k > n: return 0\n    return math.comb(n, k)"
    },
    {
        "task_id": 13,
        "prompt": "Write a function to flatten a shallow nested list of lists into a single list.",
        "test": "def flatten_list(nested_list):\n    return [item for sublist in nested_list for item in sublist]"
    },
    {
        "task_id": 14,
        "prompt": "Write a function to check if all elements in a list are unique.",
        "test": "def all_unique(lst):\n    return len(lst) == len(set(lst))"
    },
    {
        "task_id": 15,
        "prompt": "Write a python function to calculate the surface area of a cylinder.",
        "test": "import math\ndef cylinder_surface_area(r, h):\n    return 2 * math.pi * r * (r + h)"
    },
    {
        "task_id": 16,
        "prompt": "Write a function to remove consecutive duplicates from a list.",
        "test": "def remove_consecutive_duplicates(lst):\n    if not lst: return []\n    res = [lst[0]]\n    for x in lst[1:]:\n        if x != res[-1]: res.append(x)\n    return res"
    },
    {
        "task_id": 17,
        "prompt": "Write a function to check if a binary tree represented as an array is a valid min-heap.",
        "test": "def is_min_heap(arr):\n    n = len(arr)\n    for i in range((n - 2) // 2 + 1):\n        if 2*i + 1 < n and arr[i] > arr[2*i + 1]: return False\n        if 2*i + 2 < n and arr[i] > arr[2*i + 2]: return False\n    return True"
    },
    {
        "task_id": 18,
        "prompt": "Write a python function to find the maximum product of two integers in an array.",
        "test": "def max_product_two(arr):\n    arr.sort()\n    return max(arr[-1] * arr[-2], arr[0] * arr[1])"
    },
    {
        "task_id": 19,
        "prompt": "Write a function to perform run-length encoding on a string.",
        "test": "def rle_encode(s):\n    if not s: return ''\n    res = []\n    count = 1\n    for i in range(1, len(s)):\n        if s[i] == s[i-1]: count += 1\n        else:\n            res.append(f'{count}{s[i-1]}')\n            count = 1\n    res.append(f'{count}{s[-1]}')\n    return ''.join(res)"
    },
    {
        "task_id": 20,
        "prompt": "Write a python function to convert snake_case string to camelCase string.",
        "test": "def snake_to_camel(s):\n    parts = s.split('_')\n    return parts[0] + ''.join(x.capitalize() for x in parts[1:])"
    },
    {
        "task_id": 21,
        "prompt": "Write a function to determine if two rectangles overlap given their corners.",
        "test": "def rectangles_overlap(r1, r2):\n    # r = (x1, y1, x2, y2)\n    if r1[0] >= r2[2] or r2[0] >= r1[2]: return False\n    if r1[1] >= r2[3] or r2[1] >= r1[3]: return False\n    return True"
    },
    {
        "task_id": 22,
        "prompt": "Write a function to calculate the Manhattan distance between two 2D points.",
        "test": "def manhattan_distance(p1, p2):\n    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])"
    },
    {
        "task_id": 23,
        "prompt": "Write a function to find the cumulative sum of a numeric list.",
        "test": "def cumsum(lst):\n    res = []\n    s = 0\n    for x in lst:\n        s += x\n        res.append(s)\n    return res"
    },
    {
        "task_id": 24,
        "prompt": "Write a python function to rotate a list to the right by k steps.",
        "test": "def rotate_right(lst, k):\n    if not lst: return []\n    k = k % len(lst)\n    return lst[-k:] + lst[:-k]"
    },
    {
        "task_id": 25,
        "prompt": "Write a function to compute the power set of a set of elements.",
        "test": "def power_set(s):\n    elems = list(s)\n    n = len(elems)\n    res = []\n    for i in range(1 << n):\n        res.append([elems[j] for j in range(n) if (i & (1 << j))])\n    return res"
    }
]

ARC_CHALLENGE_SAMPLES = [
    {
        "id": "ARC_01",
        "question": "An astronomer observes that a distant galaxy's spectral lines are shifted toward the red end of the spectrum. What does this observation primarily indicate?",
        "choices": "(A) The galaxy is approaching Earth (B) The galaxy is receding from Earth (C) The galaxy is losing mass (D) The galaxy is composed mainly of helium",
        "answer": "(B)"
    },
    {
        "id": "ARC_02",
        "question": "Which of the following cellular organelles contains its own circular DNA molecule and ribosomes similar to those found in bacteria?",
        "choices": "(A) Endoplasmic reticulum (B) Mitochondria (C) Golgi apparatus (D) Lysosome",
        "answer": "(B)"
    },
    {
        "id": "ARC_03",
        "question": "When solid sodium chloride dissolves in water, what type of intermolecular interaction is formed between the sodium ions and water molecules?",
        "choices": "(A) Hydrogen bonding (B) Ion-dipole interaction (C) London dispersion forces (D) Covalent bonding",
        "answer": "(B)"
    },
    {
        "id": "ARC_04",
        "question": "According to Le Chatelier's principle, what will happen to an exothermic reaction at equilibrium if the temperature of the system is increased?",
        "choices": "(A) Equilibrium shifts toward the products (B) Equilibrium shifts toward the reactants (C) The equilibrium constant increases (D) The reaction ceases completely",
        "answer": "(B)"
    },
    {
        "id": "ARC_05",
        "question": "A sound wave travels from air into water. What happens to its speed and wavelength?",
        "choices": "(A) Speed decreases, wavelength decreases (B) Speed increases, wavelength increases (C) Speed increases, wavelength decreases (D) Speed decreases, wavelength increases",
        "answer": "(B)"
    },
    {
        "id": "ARC_06",
        "question": "Which layer of Earth's atmosphere contains the highest concentration of ozone that absorbs ultraviolet solar radiation?",
        "choices": "(A) Troposphere (B) Stratosphere (C) Mesosphere (D) Thermosphere",
        "answer": "(B)"
    },
    {
        "id": "ARC_07",
        "question": "In a semiconductor at absolute zero (0 Kelvin), where are the electrons located relative to the energy bands?",
        "choices": "(A) In the conduction band only (B) Completely filling the valence band with an empty conduction band (C) In the forbidden band gap (D) Distributed equally between both bands",
        "answer": "(B)"
    },
    {
        "id": "ARC_08",
        "question": "Which fundamental interaction is responsible for holding quarks together inside protons and neutrons?",
        "choices": "(A) Gravitational force (B) Electromagnetic force (C) Strong nuclear force (D) Weak nuclear force",
        "answer": "(C)"
    },
    {
        "id": "ARC_09",
        "question": "What is the primary function of the myelin sheath that surrounds the axon of a neuron?",
        "choices": "(A) To synthesize neurotransmitters (B) To increase the propagation speed of action potentials (C) To absorb toxic waste products (D) To regulate synaptic cleft gene expression",
        "answer": "(B)"
    },
    {
        "id": "ARC_10",
        "question": "A rock sample contains a radioactive isotope with a half-life of 500 million years. If only 12.5% of the original parent isotope remains, how old is the rock?",
        "choices": "(A) 500 million years (B) 1.0 billion years (C) 1.5 billion years (D) 2.0 billion years",
        "answer": "(C)"
    },
    {
        "id": "ARC_11",
        "question": "Which of the following processes represents a decrease in the entropy of the thermodynamic system?",
        "choices": "(A) Evaporation of water (B) Sublimation of dry ice (C) Freezing of liquid water into ice (D) Dissolution of sugar in hot tea",
        "answer": "(C)"
    },
    {
        "id": "ARC_12",
        "question": "What happens to the resistance of an ideal superconductor when its temperature falls below its critical temperature (Tc)?",
        "choices": "(A) It increases logarithmically (B) It drops abruptly to exactly zero (C) It oscillates sinusoidally (D) It equals Planck's constant",
        "answer": "(B)"
    },
    {
        "id": "ARC_13",
        "question": "In meiosis, during which stage does genetic crossing over (homologous recombination) occur?",
        "choices": "(A) Prophase I (B) Metaphase II (C) Anaphase I (D) Telophase II",
        "answer": "(A)"
    },
    {
        "id": "ARC_14",
        "question": "Which type of seismic wave is a transverse shear wave that cannot propagate through liquids such as Earth's outer core?",
        "choices": "(A) P-waves (B) S-waves (C) Rayleigh waves (D) Pressure waves",
        "answer": "(B)"
    },
    {
        "id": "ARC_15",
        "question": "What is the oxidation number of sulfur in the sulfate ion (SO4^2-)?",
        "choices": "(A) +2 (B) +4 (C) +6 (D) -2",
        "answer": "(C)"
    },
    {
        "id": "ARC_16",
        "question": "A light ray passes from crown glass (n=1.52) into air (n=1.00). At angles of incidence greater than the critical angle, what phenomenon occurs?",
        "choices": "(A) Complete dispersion (B) Total internal reflection (C) Constructive interference (D) Circular polarization",
        "answer": "(B)"
    },
    {
        "id": "ARC_17",
        "question": "Which cellular metabolic pathway produces the highest yield of ATP molecules per molecule of glucose oxidized?",
        "choices": "(A) Glycolysis (B) Fermentation (C) Oxidative phosphorylation (D) Citric acid cycle",
        "answer": "(C)"
    },
    {
        "id": "ARC_18",
        "question": "What is the primary greenhouse gas emitted by human activity that contributes most significantly to long-term radiative forcing?",
        "choices": "(A) Methane (B) Carbon dioxide (C) Nitrous oxide (D) Sulfur hexafluoride",
        "answer": "(B)"
    },
    {
        "id": "ARC_19",
        "question": "An ideal gas undergoes an isothermal expansion. Which statement is true regarding its internal energy (U)?",
        "choices": "(A) U increases (B) U decreases (C) U remains constant (D) U becomes zero",
        "answer": "(C)"
    },
    {
        "id": "ARC_20",
        "question": "In genetics, what term describes a single gene having multiple seemingly unrelated phenotypic effects?",
        "choices": "(A) Epistasis (B) Pleiotropy (C) Codominance (D) Incomplete penetrance",
        "answer": "(B)"
    },
    {
        "id": "ARC_21",
        "question": "Which element has the highest first ionization energy on the periodic table?",
        "choices": "(A) Fluorine (B) Helium (C) Chlorine (D) Hydrogen",
        "answer": "(B)"
    },
    {
        "id": "ARC_22",
        "question": "In Newtonian mechanics, what quantity is always conserved in both elastic and inelastic collisions between isolated objects?",
        "choices": "(A) Kinetic energy (B) Linear momentum (C) Mechanical energy (D) Angular velocity",
        "answer": "(B)"
    },
    {
        "id": "ARC_23",
        "question": "What geological boundary marks the transition between Earth's crust and the upper mantle?",
        "choices": "(A) Gutenberg discontinuity (B) Mohorovicic discontinuity (C) Lehmann discontinuity (D) Conrad discontinuity",
        "answer": "(B)"
    },
    {
        "id": "ARC_24",
        "question": "Which enzyme is responsible for unwinding the double helix during DNA replication in Escherichia coli?",
        "choices": "(A) DNA polymerase III (B) Helicase (C) Primase (D) Topoisomerase",
        "answer": "(B)"
    },
    {
        "id": "ARC_25",
        "question": "When a capacitor is connected across an AC voltage source, what is the phase relationship between current and voltage?",
        "choices": "(A) Current leads voltage by 90 degrees (B) Voltage leads current by 90 degrees (C) Current and voltage are in phase (D) Current leads voltage by 180 degrees",
        "answer": "(A)"
    }
]

MMLU_STEM_SAMPLES = [
    {
        "subject": "Computer Science",
        "question": "In computational complexity theory, what is the time complexity of the Floyd-Warshall algorithm for all-pairs shortest paths on a graph with V vertices?",
        "choices": "(A) O(V^2) (B) O(V^3) (C) O(V log V) (D) O(V * E log V)",
        "answer": "(B)"
    },
    {
        "subject": "Computer Science",
        "question": "Which synchronization primitive uses atomic test-and-set operations to enforce mutual exclusion without causing context switches?",
        "choices": "(A) Spinlock (B) Semaphore (C) Monitor (D) Condition variable",
        "answer": "(A)"
    },
    {
        "subject": "Computer Science",
        "question": "In relational database theory, a relation is in Boyce-Codd Normal Form (BCNF) if and only if for every non-trivial functional dependency X -> Y:",
        "choices": "(A) Y is a prime attribute (B) X is a superkey (C) X is an atomic attribute (D) Y contains X",
        "answer": "(B)"
    },
    {
        "subject": "Computer Science",
        "question": "Which IEEE standard specifies the format and rounding rules for binary 32-bit and 64-bit floating-point arithmetic?",
        "choices": "(A) IEEE 802.11 (B) IEEE 754 (C) IEEE 1394 (D) IEEE 488",
        "answer": "(B)"
    },
    {
        "subject": "Computer Science",
        "question": "What is the amortized time complexity of inserting an element into a dynamic array (vector) of initial capacity N when resizing by doubling?",
        "choices": "(A) O(N) (B) O(log N) (C) O(1) (D) O(N^2)",
        "answer": "(C)"
    },
    {
        "subject": "Mathematics",
        "question": "What is the eigenvalues of a 2x2 identity matrix?",
        "choices": "(A) 0 and 1 (B) 1 with algebraic multiplicity 2 (C) -1 and 1 (D) 2 and 0",
        "answer": "(B)"
    },
    {
        "subject": "Mathematics",
        "question": "What is the value of the integral from -infinity to +infinity of exp(-x^2) dx?",
        "choices": "(A) 1 (B) sqrt(pi) (C) pi (D) 2 * sqrt(pi)",
        "answer": "(B)"
    },
    {
        "subject": "Mathematics",
        "question": "According to Bayes' theorem, if P(A) = 0.2, P(B|A) = 0.8, and P(B) = 0.4, what is the posterior probability P(A|B)?",
        "choices": "(A) 0.16 (B) 0.40 (C) 0.50 (D) 0.25",
        "answer": "(B)"
    },
    {
        "subject": "Mathematics",
        "question": "In group theory, what is the order of the alternating group A_4?",
        "choices": "(A) 4 (B) 12 (C) 24 (D) 6",
        "answer": "(B)"
    },
    {
        "subject": "Mathematics",
        "question": "Which of the following functions is analytic everywhere on the complex plane (an entire function)?",
        "choices": "(A) f(z) = 1/z (B) f(z) = exp(z) (C) f(z) = ln(z) (D) f(z) = conj(z)",
        "answer": "(B)"
    },
    {
        "subject": "Physics",
        "question": "In quantum mechanics, what is the commutator [x, p_x] of position and linear momentum operators?",
        "choices": "(A) 0 (B) i * hbar (C) -i * hbar (D) hbar / 2",
        "answer": "(B)"
    },
    {
        "subject": "Physics",
        "question": "What does Maxwell's displacement current term in Ampere's law account for?",
        "choices": "(A) Static electric fields (B) Time-varying electric fields producing magnetic fields (C) Conservation of thermal energy (D) Superconducting Cooper pairs",
        "answer": "(B)"
    },
    {
        "subject": "Physics",
        "question": "In special relativity, as the velocity of an object with rest mass m approaches the speed of light c, its total energy:",
        "choices": "(A) Approaches mc^2 (B) Approaches zero (C) Approaches infinity (D) Oscillates between kinetic and potential",
        "answer": "(C)"
    },
    {
        "subject": "Physics",
        "question": "What type of statistical distribution governs identical, indistinguishable particles with half-integer spin (fermions)?",
        "choices": "(A) Bose-Einstein statistics (B) Fermi-Dirac statistics (C) Maxwell-Boltzmann statistics (D) Poisson distribution",
        "answer": "(B)"
    },
    {
        "subject": "Physics",
        "question": "What is the de Broglie wavelength of a particle with linear momentum p?",
        "choices": "(A) lambda = h / p (B) lambda = p / h (C) lambda = h * p (D) lambda = hbar * p",
        "answer": "(A)"
    },
    {
        "subject": "Chemistry",
        "question": "What is the hybridization state of the carbon atom in acetylene (ethyne, C2H2)?",
        "choices": "(A) sp3 (B) sp2 (C) sp (D) dsp2",
        "answer": "(C)"
    },
    {
        "subject": "Chemistry",
        "question": "Which thermodynamic state function represents the maximum non-expansion work obtainable from a closed system at constant temperature and pressure?",
        "choices": "(A) Enthalpy (H) (B) Helmholtz free energy (A) (C) Gibbs free energy (G) (D) Internal energy (U)",
        "answer": "(C)"
    },
    {
        "subject": "Chemistry",
        "question": "According to the Bronsted-Lowry definition, what is a conjugate acid?",
        "choices": "(A) The species formed when a base gains a proton (B) An electron-pair acceptor (C) The species formed when an acid loses a proton (D) A substance that dissociates OH- ions",
        "answer": "(A)"
    },
    {
        "subject": "Chemistry",
        "question": "In coordination chemistry, what is the coordination number of the central cobalt ion in [Co(NH3)6]^3+?",
        "choices": "(A) 3 (B) 6 (C) 9 (D) 12",
        "answer": "(B)"
    },
    {
        "subject": "Chemistry",
        "question": "Which of the following organic reactions converts a ketone into a secondary alcohol via hydride reduction?",
        "choices": "(A) Reaction with NaBH4 (B) Reaction with KMnO4 (C) Wolff-Kishner reduction (D) Ozonolysis",
        "answer": "(A)"
    },
    {
        "subject": "Computer Science",
        "question": "Which sorting algorithm has a worst-case time complexity of O(N log N) while being in-place but unstable?",
        "choices": "(A) QuickSort (B) HeapSort (C) MergeSort (D) BubbleSort",
        "answer": "(B)"
    },
    {
        "subject": "Mathematics",
        "question": "What is the divergence of the curl of any twice continuously differentiable vector field F in R^3?",
        "choices": "(A) 1 (B) 0 (C) div(F) (D) laplacian(F)",
        "answer": "(B)"
    },
    {
        "subject": "Physics",
        "question": "In thermodynamics, what does the Second Law dictate about the entropy of an isolated system undergoing any spontaneous process?",
        "choices": "(A) It must strictly decrease (B) It can never decrease (C) It remains constant at zero (D) It fluctuates symmetrically",
        "answer": "(B)"
    },
    {
        "subject": "Chemistry",
        "question": "Which principle states that no two electrons in an atom can have the same set of four quantum numbers?",
        "choices": "(A) Hund's rule (B) Pauli exclusion principle (C) Aufbau principle (D) Heisenberg uncertainty principle",
        "answer": "(B)"
    },
    {
        "subject": "Computer Science",
        "question": "In networking, which layer of the OSI 7-layer reference model is responsible for logical IP addressing and path routing?",
        "choices": "(A) Data Link Layer (B) Network Layer (C) Transport Layer (D) Session Layer",
        "answer": "(B)"
    }
]

HELLASWAG_SAMPLES = [
    {
        "activity": "Roofing repair",
        "context": "A man climbs up a ladder carrying a bundle of asphalt shingles on his shoulder. He reaches the pitched roof and sets the shingles down near the chimney.",
        "choices": "(A) He sits down on the ladder and starts eating his lunch. (B) He opens the bundle with a utility knife and begins laying out shingles along the chalk line. (C) He sets the shingles on fire to melt the tar immediately. (D) He throws the ladder into the garden and jumps off the roof.",
        "answer": "(B)"
    },
    {
        "activity": "Baking bread",
        "context": "The baker kneads the dough vigorously on a floured countertop until it becomes smooth and elastic. She places the dough ball into a lightly oiled bowl.",
        "choices": "(A) She immediately slices it into thin croutons and eats it raw. (B) She covers the bowl with a damp cloth and leaves it in a warm area to proof and rise. (C) She throws the bowl out of the window into the street. (D) She fills the bowl with vinegar and scrubs the floor.",
        "answer": "(B)"
    },
    {
        "activity": "Playing chess",
        "context": "White advances a knight into the center of the board, threatening Black's undefended bishop. Black studies the board carefully for several minutes.",
        "choices": "(A) Black removes the king and hides it in their pocket. (B) Black retreats the bishop to a safe square while reinforcing their pawn structure. (C) Black overturns the chessboard and walks away in anger. (D) Black places three checkers on the queen's square.",
        "answer": "(B)"
    },
    {
        "activity": "Car maintenance",
        "context": "The mechanic slides a drain pan underneath the engine oil pan. Using a wrench, he loosens the drain plug carefully.",
        "choices": "(A) He sprays water into the gas tank until it overflows. (B) Dark used motor oil flows steadily into the drain pan as he removes the plug. (C) The engine explodes instantly in a cloud of pink smoke. (D) He paints the tires bright yellow with a broad brush.",
        "answer": "(B)"
    },
    {
        "activity": "Playing guitar",
        "context": "The guitarist plugs the electric guitar into an amplifier, turns up the gain knob, and steps on a distortion pedal.",
        "choices": "(A) He plucks the low E string, unleashing a loud sustained power chord through the speakers. (B) He cuts all the guitar strings with scissors and drinks a glass of milk. (C) The amplifier turns into an acoustic violin. (D) He whispers silently into the soundhole without touching the instrument.",
        "answer": "(A)"
    },
    {
        "activity": "Gardening",
        "context": "A woman digs a small trench in the fertile garden soil during early spring. She spaces tomato seedlings eight inches apart inside the trench.",
        "choices": "(A) She pours boiling saltwater directly on the roots to sterilize them. (B) She gently packs the rich soil around each seedling's base and waters them thoroughly. (C) She covers the garden in cement to prevent weeds from ever growing. (D) She pulls up the fence and runs down the road.",
        "answer": "(B)"
    },
    {
        "activity": "Archery practice",
        "context": "The archer stands perpendicular to the target, notches an arrow onto the bowstring, and raises the recurve bow.",
        "choices": "(A) She drops the bow and starts running in circles around the range. (B) She draws the string smoothly to her anchor point on her cheek, takes steady aim, and releases. (C) She ties the arrow around her wrist and reads a novel. (D) She breaks the arrow in half and feeds it to a bird.",
        "answer": "(B)"
    },
    {
        "activity": "Painting a room",
        "context": "The painter covers the wooden baseboards and window trims with blue painter's tape. He pours matte latex paint into a roller tray.",
        "choices": "(A) He dips the roller evenly into the paint and rolls an overlapping 'W' pattern onto the drywall. (B) He drinks the paint from the tray with a plastic straw. (C) He sets fire to the drywall with a blowtorch. (D) He throws the roller through the closed windowpane.",
        "answer": "(A)"
    },
    {
        "activity": "Laboratory experiment",
        "context": "The chemist measures ten milliliters of hydrochloric acid in a graduated cylinder under the fume hood. She brings the cylinder toward an Erlenmeyer flask containing zinc granules.",
        "choices": "(A) She slowly pours the acid over the zinc, and effervescence begins immediately as hydrogen gas bubbles evolve. (B) She drinks the hydrochloric acid and laughs loudly. (C) The glass flask turns into a flock of wild doves. (D) She pours the acid directly onto her bare skin.",
        "answer": "(A)"
    },
    {
        "activity": "Coffee brewing",
        "context": "The barista grinds fresh roasted espresso beans into the portafilter until it is slightly mounded. He uses a metal tamper to press the coffee grounds firmly.",
        "choices": "(A) He pours orange juice into the grounds and mixes them with a fork. (B) He wipes the portafilter rim clean and locks it securely into the espresso group head. (C) He throws the coffee grounds into the ceiling fan. (D) He places the portafilter in a microwave oven for twenty minutes.",
        "answer": "(B)"
    },
    {
        "activity": "Bicycle repair",
        "context": "The cyclist flips the bicycle upside down on the lawn to fix a flat rear tire. She inserts tire levers under the tire bead to pry it off the rim.",
        "choices": "(A) She removes the punctured inner tube to locate the leak in a basin of water. (B) She sets fire to the rubber tire to vulcanize the patch instantly. (C) She replaces the handlebars with wooden branches from an oak tree. (D) She removes the pedals and places them in her ears.",
        "answer": "(A)"
    },
    {
        "activity": "Woodworking",
        "context": "The carpenter measures twice along a pine board and marks a straight line with a carpenter's pencil. He clamps the board securely to his workbench.",
        "choices": "(A) He aligns the circular saw blade with the waste side of the pencil line and makes a smooth, steady cut. (B) He chews on the wood with his teeth until it splits. (C) He paints the pencil line with thick strawberry syrup. (D) He throws the workbench into the fireplace.",
        "answer": "(A)"
    },
    {
        "activity": "Cooking steak",
        "context": "The chef heats a heavy cast-iron skillet over high flame until light wisps of smoke appear. He places a well-seasoned ribeye steak directly onto the hot surface.",
        "choices": "(A) The steak sizzles loudly as a deep brown caramelized crust begins to form. (B) He immediately pours a gallon of cold ice water into the sizzling pan. (C) He removes the steak and feeds it raw to a goldfish. (D) He takes the pan off the stove and places it in the clothes dryer.",
        "answer": "(A)"
    },
    {
        "activity": "Pottery throwing",
        "context": "The potter centers a lump of wet stoneware clay on the rotating potter's wheel. She wets her hands and pushes her thumbs into the exact center of the revolving clay.",
        "choices": "(A) She creates an opening and gradually pulls the walls upward to form a symmetrical cylinder. (B) She stops the wheel and hits the clay with a heavy sledgehammer. (C) She eats the clay with a spoon and drinks saltwater. (D) She paints the clay with red house paint while it rotates.",
        "answer": "(A)"
    },
    {
        "activity": "Fishing",
        "context": "The angler spots a rising trout near the bank of a calm mountain river. He ties a small dry fly to the fine tippet and steps into the shallow water.",
        "choices": "(A) He makes two false casts to measure the distance and delivers the fly softly onto the surface upstream of the fish. (B) He drops his rod into the mud and shouts at the trees. (C) He tosses dynamite into the river and laughs uncontrollably. (D) He drinks the river water through his fishing net.",
        "answer": "(A)"
    },
    {
        "activity": "Camping fire building",
        "context": "In the clearing, the camper arranges dry tinder shavings inside a loose tepee structure of thin kindling sticks. He prepares a box of waterproof matches.",
        "choices": "(A) He strikes a match, shields the flame with his hand, and ignites the tinder at its base until small flames lick the kindling. (B) He empties an entire fire extinguisher over the dry wood. (C) He covers the kindling with heavy wet mud from the riverbank. (D) He places a battery inside the kindling and waits for it to freeze.",
        "answer": "(A)"
    },
    {
        "activity": "Playing soccer",
        "context": "The forward receives a through ball behind the opponent's defensive line, leaving only the goalkeeper between him and the net. The goalkeeper rushes forward to narrow the shooting angle.",
        "choices": "(A) The forward smoothly chips the ball over the sliding goalkeeper toward the open goal. (B) The forward picks up the ball with both hands and runs into the stands. (C) The forward falls asleep on the grass immediately. (D) The forward sits down and starts untying his shoes.",
        "answer": "(A)"
    },
    {
        "activity": "Sewing a button",
        "context": "The tailor threads a needle with strong matching thread and ties a small knot at the end. He positions the missing four-hole button over the fabric.",
        "choices": "(A) He passes the needle up through the fabric and through one hole of the button, sewing repeated diagonal stitches to secure it. (B) He hammers a heavy steel nail straight through the button into the table. (C) He burns the button with a lighter until it dissolves into ash. (D) He swallows the button with a cup of cold water.",
        "answer": "(A)"
    },
    {
        "activity": "Playing piano",
        "context": "The pianist sits on the bench with upright posture, opens the sheet music to Chopin's Nocturne in E-flat major, and adjusts the height of the stool.",
        "choices": "(A) He places his hands gently on the keys and begins playing the expressive lyrical melody with delicate phrasing. (B) He slams the fallboard down violently onto his fingers. (C) He pours a cup of hot soup directly onto the piano keys. (D) He cuts the piano strings with wire cutters.",
        "answer": "(A)"
    },
    {
        "activity": "Washing dishes",
        "context": "After dinner, the sink is filled with hot soapy water. She grabs a greasy frying pan and a non-abrasive sponge.",
        "choices": "(A) She scrubs the interior surface of the pan in circular motions to break down the grease before rinsing with clean water. (B) She places the dirty pan on top of the living room television. (C) She throws the pan out into the backyard fence. (D) She fills the pan with sugar and feeds it to the ants.",
        "answer": "(A)"
    },
    {
        "activity": "Surfing",
        "context": "The surfer paddles out past the breaking surf, sitting upright on his surfboard while watching the oncoming set of waves. A smooth cresting shoulder approaches.",
        "choices": "(A) He turns the board toward the shore, paddles hard with both arms to match the wave's speed, and pops up onto his feet as it catches him. (B) He unties his leash and throws his board into the open ocean. (C) He dives straight to the seafloor and buries his head in the sand. (D) He takes out a notebook and starts writing an essay.",
        "answer": "(A)"
    },
    {
        "activity": "Changing a lightbulb",
        "context": "The ceiling light in the hallway flickers and goes completely dark. The homeowner turns off the light wall switch and brings an A-frame stepladder.",
        "choices": "(A) He climbs the ladder, unscrews the burned-out bulb counter-clockwise, and screws in a new LED replacement until snug. (B) He smashes the ceiling fixture with a wooden baseball bat. (C) He sprays lighter fluid into the empty light socket. (D) He removes the wall switch and puts it in the refrigerator.",
        "answer": "(A)"
    },
    {
        "activity": "Origami folding",
        "context": "The student places a crisp square of colored kami paper on the smooth table with the colored side facing downward. He folds the paper diagonally in half from corner to corner.",
        "choices": "(A) He creases the folded edge firmly with the edge of his thumbnail to create a sharp triangular base. (B) He tears the paper into twenty shredded strips and scatters them. (C) He sets the paper floating in a cup of orange juice. (D) He glues the paper to the bottom of his shoe.",
        "answer": "(A)"
    },
    {
        "activity": "Ironing a shirt",
        "context": "The steam iron heats up to the cotton temperature setting on the ironing board. He lays the collar of a wrinkled dress shirt flat against the board.",
        "choices": "(A) He presses the hot iron firmly from the points toward the center of the collar, releasing steam to remove the creases. (B) He pours cold paint over the shirt and hangs it on a tree. (C) He places the iron inside the pocket of the shirt and walks away. (D) He cuts the sleeves off the shirt with heavy shears.",
        "answer": "(A)"
    },
    {
        "activity": "Playing tennis",
        "context": "The server bounces the tennis ball twice behind the baseline, focuses on the service box, and tosses the ball smoothly into the air above his hitting shoulder.",
        "choices": "(A) He extends his racquet fully at the apex of the toss and drives the ball across the net with a crisp flat serve. (B) He catches the ball and drops the racquet onto his foot. (C) He hits the ball backward into the spectators' seats. (D) He jumps over the net and argues with the empty court.",
        "answer": "(A)"
    }
]
