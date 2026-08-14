from __future__ import annotations

from gold_dsl import Case, Step, claim, definition, target


CASES = (
    Case(
        source_id="cmimc_2025/16",
        steps=(
            Step("There are $3 \\times 3 = 9$ such $2 \\times 2$ subgrids in a $4 \\times 4$ grid."),
            Step("We find that there are **exactly three** configurations that use **4 coins** and satisfy both constraints:"),
            Step("$$\n2^4 = 16 \\text{ configurations for each 4-coin base}\n$$"),
            Step("$$\n\\text{Total configurations} = 3 \\times 16 = 48\n$$"),
            Step("$$\n\\text{Final count} = 48 - 2 = 46\n$$"),
        ),
        nodes=(
            definition("checkerCover46", "formal_bridge", "def checkerCover46 (m : Fin 256) : Bool := (m.val.testBit 0 || m.val.testBit 2) && (m.val.testBit 1 || m.val.testBit 2) && (m.val.testBit 1 || m.val.testBit 3) && (m.val.testBit 2 || m.val.testBit 4) && (m.val.testBit 2 || m.val.testBit 5) && (m.val.testBit 3 || m.val.testBit 5) && (m.val.testBit 4 || m.val.testBit 6) && (m.val.testBit 5 || m.val.testBit 6) && (m.val.testBit 5 || m.val.testBit 7)", source_steps=(1, 2, 3), statement="Subsets of one checkerboard color automatically avoid adjacency; the nine clauses cover all 2x2 blocks"),
            definition("checkerCoverSet46", "formal_bridge", "def checkerCoverSet46 : Finset (Fin 256) := Finset.univ.filter (fun m => checkerCover46 m = true)", source_steps=(1, 2, 3)),
            claim("checker_family_count46", "lemma checker_family_count46 : checkerCoverSet46.card = 52", role="formal_bridge", source_steps=(1, 2, 3), label="proved", proof="by decide", method="enumerate_256_checker_subsets"),
            target(
                "gold_cmimc_2025_16", "theorem gold_cmimc_2025_16 : checkerCoverSet46.card = 46",
                dependencies=("checker_family_count46",), source_steps=(4, 5), label="disproved",
                proof="by rw [checker_family_count46]; norm_num", method="already_52_in_one_checkerboard_family",
                statement="A single valid checkerboard-side family already contains 52 configurations, contradicting the total 46",
            ),
        ),
    ),
    Case(
        source_id="hmmt_feb_2025/12",
        steps=(
            Step("$$\n\\binom{7}{4} = 35\n$$"),
            Step("We are interested in the **number of those 35 removals** that result in **no path** from $(1,1)$ to $(3,3)$"),
            Step("After analyzing all possible configurations and considering the structure of the grid, we find that the number of **ways to remove 4 lily pads** such that **no path exists** from $(1,1)$ to $(3,3)$ is:\n\n$$\n\\boxed{16}\n$$"),
            Step("$$\n\\boxed{16}\n$$"),
        ),
        nodes=(
            definition("isolateStartCount12", "formal_bridge", "def isolateStartCount12 : ℕ := Nat.choose 5 2", source_steps=(1, 2, 3), statement="Remove both neighbors of the start and any two of the remaining five removable pads"),
            definition("isolateTargetCount12", "formal_bridge", "def isolateTargetCount12 : ℕ := Nat.choose 5 2", source_steps=(1, 2, 3), statement="Symmetric family isolating the target"),
            claim("two_isolation_families12", "lemma two_isolation_families12 : isolateStartCount12 + isolateTargetCount12 - 1 = 19", role="formal_bridge", source_steps=(1, 2, 3), label="proved", proof="by norm_num [isolateStartCount12, isolateTargetCount12, Nat.choose]", method="two_ten_element_families_one_overlap"),
            target(
                "gold_hmmt_feb_2025_12", "theorem gold_hmmt_feb_2025_12 : isolateStartCount12 + isolateTargetCount12 - 1 = 16",
                dependencies=("two_isolation_families12",), source_steps=(3, 4), label="disproved",
                proof="by norm_num [isolateStartCount12, isolateTargetCount12, Nat.choose]", method="nineteen_explicit_disconnectors",
                statement="Two elementary isolation families already give nineteen distinct disconnecting removals",
            ),
        ),
    ),
    Case(
        source_id="hmmt_feb_2025/14",
        steps=(
            Step("Sophie can move in four directions:"),
            Step("She **cannot revisit any points** along her path."),
            Step("After carefully simulating the DFS approach and counting all valid paths that meet the constraints (no revisiting, staying within bounds), we find that the total number of valid paths from $(0,0)$ to $(3,3)$ is:"),
            Step("$$\n\\boxed{12}\n$$"),
        ),
        nodes=(
            definition("monotonePathCount14", "formal_bridge", "def monotonePathCount14 : ℕ := Nat.choose 6 3", source_steps=(1, 2, 3), statement="Every ordering of three right and three up moves is a valid non-revisiting path"),
            claim("twenty_monotone_paths14", "lemma twenty_monotone_paths14 : monotonePathCount14 = 20", role="formal_bridge", source_steps=(1, 2, 3), label="proved", proof="by norm_num [monotonePathCount14, Nat.choose]", method="choose_positions_of_three_right_moves"),
            target(
                "gold_hmmt_feb_2025_14", "theorem gold_hmmt_feb_2025_14 : monotonePathCount14 = 12",
                dependencies=("twenty_monotone_paths14",), source_steps=(3, 4), label="disproved",
                proof="by norm_num [monotonePathCount14, Nat.choose]", method="twenty_valid_monotone_paths_already",
                statement="The allowed move set contains the twenty ordinary monotone paths, so the total cannot be twelve",
            ),
        ),
    ),
    Case(
        source_id="hmmt_feb_2025/16",
        steps=(
            Step("$$\n\\binom{6}{2}^2 = 15^2 = 225\n$$"),
            Step("$$\n\\binom{225}{2} = \\frac{225 \\times 224}{2} = 25200\n$$"),
            Step("$$\nS = 26600 \\quad \\Rightarrow \\quad N = \\frac{26600}{2} = 13300\n$$"),
            Step("$$\n\\boxed{13300}\n$$"),
        ),
        nodes=(
            definition("xSeparatedPairs16", "formal_bridge", "def xSeparatedPairs16 : ℕ := Nat.choose 6 4 * (Nat.choose 6 2)^2", source_steps=(1, 2, 3)),
            definition("bothSeparatedPairs16", "formal_bridge", "def bothSeparatedPairs16 : ℕ := 2 * (Nat.choose 6 4)^2", source_steps=(1, 2, 3)),
            claim("nonoverlap_inclusion_exclusion16", "lemma nonoverlap_inclusion_exclusion16 : 2 * xSeparatedPairs16 - bothSeparatedPairs16 = 6300", role="formal_bridge", source_steps=(1, 2, 3), label="proved", proof="by norm_num [xSeparatedPairs16, bothSeparatedPairs16, Nat.choose]", method="separate_in_x_or_y_subtract_both"),
            target(
                "gold_hmmt_feb_2025_16", "theorem gold_hmmt_feb_2025_16 : 2 * xSeparatedPairs16 - bothSeparatedPairs16 = 13300",
                dependencies=("nonoverlap_inclusion_exclusion16",), source_steps=(3, 4), label="disproved",
                proof="by norm_num [xSeparatedPairs16, bothSeparatedPairs16, Nat.choose]", method="exact_closed_rectangle_separation_count",
                statement="The correct unordered count is 6300, not the unsupported ordered sum 26600 divided by two",
            ),
        ),
    ),
    Case(
        source_id="cmimc_2025/30",
        steps=(
            Step("$$\nBL = \\frac{28s + 36}{\\sqrt{100s^2 - 72s + 36}}.\n$$"),
            Step("### Step 4: Try $ s = 0.5 $ (Midpoint of $ AC $)"),
            Step("Let’s test $ s = 0.5 $, which corresponds to $ K $ being the **midpoint** of $ AC $"),
            Step("So, $ BL = 10 $."),
            Step("$$\n\\boxed{10.00}\n$$"),
        ),
        nodes=(
            definition("blSquared30", "formal_bridge", "def blSquared30 (s : ℚ) : ℚ := (28*s+36)^2 / (100*s^2-72*s+36)", source_steps=(1, 2, 3, 4)),
            claim("midpoint_value30", "lemma midpoint_value30 : blSquared30 (1/2) = 100", source_steps=(1, 2, 3, 4), label="proved", proof="by norm_num [blSquared30]", method="substitute_midpoint"),
            claim(
                "cot_trial_is_forced30", "lemma cot_trial_is_forced30 : ∀ s : ℚ, 0 ≤ s → s ≤ 1 → blSquared30 s = 100",
                dependencies=("midpoint_value30",), source_steps=(2, 3, 4), label="disproved",
                proof="by\n  push_neg\n  exact ⟨0, by norm_num, by norm_num, by norm_num [blSquared30]⟩",
                method="s_zero_distinguishes_trial_from_derivation",
                statement="Trying the midpoint verifies one value but does not derive that the geometric K is the midpoint",
            ),
            target("gold_cmimc_2025_30", "theorem gold_cmimc_2025_30 : blSquared30 (1/2) = 100", dependencies=("cot_trial_is_forced30",), source_steps=(4, 5), label="blocked_by_dependency", statement="The boxed ten rests on an unforced trial value"),
        ),
    ),
    Case(
        source_id="hmmt_feb_2025/24",
        steps=(
            Step("$$\n|c_2 - c_1| = 1\n$$"),
            Step("$$\nr_2 - 1 = 1 \\Rightarrow r_2 = 2\n$$"),
            Step("$$\nAC \\cdot CB = 2r \\sqrt{R^2 - r^2}\n$$"),
            Step("$$\nAC \\cdot CB = 2 \\cdot 2 \\cdot \\sqrt{10^2 - 2^2} = 4 \\cdot \\sqrt{100 - 4} = 4 \\cdot \\sqrt{96} = 4 \\cdot 4\\sqrt{6} = 16\\sqrt{6}\n$$"),
            Step("$$\n\\boxed{16\\sqrt{6}}\n$$"),
        ),
        nodes=(
            definition("centerDistance24", "cot_claim", "def centerDistance24 : ℚ := 1", source_steps=(1,)),
            claim(
                "cot_center_distance_forces_radius24", "lemma cot_center_distance_forces_radius24 : ∀ r : ℚ, centerDistance24 = 1 → r - 1 = 1",
                source_steps=(1, 2), label="disproved",
                proof="by\n  push_neg\n  exact ⟨3, by norm_num [centerDistance24], by norm_num⟩",
                method="center_distance_contains_no_radius_equation",
                statement="The displayed center separation alone does not imply r2 minus one equals one",
            ),
            target(
                "gold_hmmt_feb_2025_24", "theorem gold_hmmt_feb_2025_24 : (2 : ℚ) * 2 * (100 - 2^2) = 384",
                dependencies=("cot_center_distance_forces_radius24",), source_steps=(3, 4, 5), label="blocked_by_dependency",
                statement="The final radical calculation depends on the unproved r2=2 bridge",
            ),
        ),
    ),
)
