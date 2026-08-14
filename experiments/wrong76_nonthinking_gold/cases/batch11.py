from __future__ import annotations

from gold_dsl import Case, Step, claim, definition, target


CASES = (
    Case(
        source_id="aime_2024/85",
        steps=(
            Step("Since the hexagon is **equilateral**, all its sides have the same length, say **s**."),
            Step("The triangle with side lengths 200, 240, and 300 lies in the plane."),
            Step("Using this setup and the relationships between the sides and angles, we can derive the following key equation from the geometric constraints:\n\n$$\ns = \\frac{1200}{17}\n$$"),
            Step("$$\n\\boxed{\\dfrac{1200}{17}}\n$$"),
        ),
        nodes=(
            definition(
                "hexSide85", "formal_bridge", "def hexSide85 : ℚ := 80",
                source_steps=(1, 2, 3),
                statement="For an equilateral opposite-parallel hexagon, the three extension-triangle side reciprocals sum to the reciprocal hexagon side; for 200, 240, 300 this gives 80",
            ),
            claim(
                "extension_triangle_relation85",
                "lemma extension_triangle_relation85 : hexSide85 * (240*300 + 200*300 + 200*240) = 200*240*300",
                role="formal_bridge", source_steps=(1, 2, 3), label="proved",
                proof="by norm_num [hexSide85]", method="reciprocal_side_relation",
            ),
            target(
                "gold_aime_2024_85", "theorem gold_aime_2024_85 : hexSide85 = 1200 / 17",
                dependencies=("extension_triangle_relation85",), source_steps=(3, 4), label="disproved",
                proof="by norm_num [hexSide85]", method="solve_extension_triangle_relation",
                statement="The extension-triangle relation yields side length 80, not 1200/17",
            ),
        ),
        fidelity_notes=("The vector-geometry reduction is isolated in one formal_bridge; the active arithmetic theorem is deliberately linear and small-model friendly.",),
    ),
    Case(
        source_id="brumo_2025/30",
        steps=(
            Step("\\frac{k \\cdot M}{(k + 1)^2} = 30"),
            Step("From Menelaus and the above, we find that $ x_E = \\pm 3 $, and $ EB \\cdot EC = 8 $, leading to $ x_E^2 - 1 = 8 \\Rightarrow x_E^2 = 9 \\Rightarrow x_E = \\pm 3 $"),
            Step("From the earlier relation $ k = 2 $, and $ kM = 30(k + 1)^2 $, we solve:"),
            Step("2M = 30 \\cdot 9 \\Rightarrow M = 135 \\Rightarrow h^2 = 134"),
            Step("$$\n\\boxed{\\sqrt{143}}\n$$"),
        ),
        nodes=(
            claim(
                "relation_allows_k_one30", "lemma relation_allows_k_one30 : (1 : ℚ) * 120 / (1 + 1)^2 = 30",
                role="formal_bridge", source_steps=(1,), label="proved", proof="by norm_num", method="explicit_alternative_solution",
            ),
            claim(
                "cot_relation_forces_k_two30",
                "lemma cot_relation_forces_k_two30 : ∀ k M : ℚ, 0 < k → 0 < M → k * M / (k + 1)^2 = 30 → k = 2",
                dependencies=("relation_allows_k_one30",), source_steps=(1, 2, 3), label="disproved",
                proof="by\n  push_neg\n  exact ⟨1, 120, by norm_num, by norm_num, by norm_num, by norm_num⟩",
                method="k_one_M_120_counterexample",
                statement="The displayed scalar relation does not establish the subsequently invoked k=2",
            ),
            target(
                "gold_brumo_2025_30", "theorem gold_brumo_2025_30 : (9 : ℕ) + 134 = 143",
                dependencies=("cot_relation_forces_k_two30",), source_steps=(3, 4, 5), label="blocked_by_dependency",
                statement="The final radical arithmetic depends on the missing/refuted k=2 inference",
            ),
        ),
    ),
    Case(
        source_id="cmimc_2025/21",
        steps=(
            Step("$$\n\\text{Area of square} = 7 \\times 7 = 49\n$$"),
            Step("Therefore, the **maximum possible surface area of the tetrahedron** is equal to the **maximum area of a parallelogram** that can be cut from the square."),
            Step("This is achieved when the **entire square is used as the net**."),
            Step("$$\n\\text{Surface Area} = 49\n$$"),
            Step("$$\n\\boxed{49}\n$$"),
        ),
        nodes=(
            definition("tetraNetShortSideSq21", "formal_bridge", "def tetraNetShortSideSq21 : ℕ := 1", source_steps=(1, 2, 3)),
            definition("tetraNetLongSideSq21", "formal_bridge", "def tetraNetLongSideSq21 : ℕ := 4 * tetraNetShortSideSq21", source_steps=(1, 2, 3), statement="A parallelogram tetrahedron net made from four congruent equilateral faces has adjacent side lengths in ratio 2:1"),
            claim(
                "tetra_net_not_square21", "lemma tetra_net_not_square21 : tetraNetLongSideSq21 ≠ tetraNetShortSideSq21",
                role="formal_bridge", source_steps=(2, 3), label="proved", proof="by norm_num [tetraNetLongSideSq21, tetraNetShortSideSq21]", method="side_ratio_obstruction",
            ),
            claim(
                "cot_entire_square_is_net21", "lemma cot_entire_square_is_net21 : tetraNetLongSideSq21 = tetraNetShortSideSq21",
                dependencies=("tetra_net_not_square21",), source_steps=(3,), label="disproved",
                proof="by norm_num [tetraNetLongSideSq21, tetraNetShortSideSq21]", method="square_cannot_have_net_side_ratio",
            ),
            target(
                "gold_cmimc_2025_21", "theorem gold_cmimc_2025_21 : (7 : ℕ) * 7 = 49",
                dependencies=("cot_entire_square_is_net21",), source_steps=(1, 4, 5), label="blocked_by_dependency",
                statement="The arithmetic area of the sheet is 49, but identifying the whole square with a valid tetrahedron net has been refuted",
            ),
        ),
    ),
    Case(
        source_id="cmimc_2025/27",
        steps=(
            Step("Thus, $ AF $ is vertical, and any point $ P $ on $ AF $ has coordinates:\n\n$$\nP = \\left(\\frac{10}{7}, \\frac{20\\sqrt{3}}{7} - t\\right)\n$$\n\nwhere $ t = PA $."),
            Step("The key condition is that $ X $, $ Y $, and $ A $ are collinear. This imposes a geometric constraint on the value of $ t = PA $."),
            Step("Using the geometric and algebraic properties of the circles and their intersections, we derive an equation involving $ t $, which ultimately simplifies to a quadratic equation in $ t $."),
            Step("After careful algebraic manipulation, we find that the only value of $ t $ that satisfies the collinearity condition is:\n\n$$\nt = \\frac{10\\sqrt{3}}{7}\n$$"),
            Step("$$\n\\boxed{\\dfrac{10\\sqrt{3}}{7}}\n$$"),
        ),
        nodes=(
            claim(
                "another_altitude_parameter27", "lemma another_altitude_parameter27 : (0 : ℚ) < 5 ∧ 5 < 20",
                role="formal_bridge", source_steps=(1,), label="proved", proof="by norm_num", method="explicit_interior_altitude_parameter",
            ),
            claim(
                "cot_displayed_constraints_force_t27", "lemma cot_displayed_constraints_force_t27 : ∀ q : ℚ, 0 < q → q < 20 → q = 10",
                dependencies=("another_altitude_parameter27",), source_steps=(1, 2, 3, 4), label="disproved",
                proof="by\n  push_neg\n  exact ⟨5, by norm_num, by norm_num, by norm_num⟩",
                method="q_five_exposes_omitted_collinearity_equation",
                statement="After scaling t=q*sqrt(3)/7, the displayed interval does not force q=10; the claimed quadratic is absent from the COT",
            ),
            target(
                "gold_cmimc_2025_27", "theorem gold_cmimc_2025_27 : (10 : ℚ) / 7 = 10 / 7",
                dependencies=("cot_displayed_constraints_force_t27",), source_steps=(4, 5), label="blocked_by_dependency",
                statement="The printed radical is not independently derived after the omitted collinearity equation",
            ),
        ),
        fidelity_notes=("This record judges the written non-thinking COT: it does not pretend that an omitted quadratic has been supplied.",),
    ),
    Case(
        source_id="cmimc_2025/38",
        steps=(
            Step("After a detailed analysis, it turns out that for complex $ m $, the only possible values that satisfy the concyclicity condition are those that lie on the **parabola** defined by $ b^2 = a + 15 $, where $ m = a + bi $."),
            Step("Therefore, we **exclude** the complex $ m $ cases, since they would introduce infinite-length curves, which contradicts the problem's statement that $ U $ is a **finite** union of curves."),
            Step("The perimeter is the **length of the line segment**, since a single point has **no length**."),
            Step("$$\n\\text{Length} = 5 - (-5) = 10\n$$"),
            Step("$$\n\\boxed{10}\n$$"),
        ),
        nodes=(
            claim(
                "nonreal_parabola_witness38", "lemma nonreal_parabola_witness38 : (1 : ℤ)^2 = -14 + 15 ∧ (1 : ℤ) ≠ 0",
                role="formal_bridge", source_steps=(1,), label="proved", proof="by norm_num", method="a_minus14_b_one",
            ),
            claim(
                "cot_parabola_has_no_nonreal_points38", "lemma cot_parabola_has_no_nonreal_points38 : ∀ a b : ℤ, b^2 = a + 15 → b = 0",
                dependencies=("nonreal_parabola_witness38",), source_steps=(1, 2), label="disproved",
                proof="by\n  push_neg\n  exact ⟨-14, 1, by norm_num, by norm_num⟩",
                method="explicit_nonreal_parabola_parameter",
                statement="Being a finite union of curves does not justify deleting an unbounded nonreal curve; the stated parabola has nonreal parameters",
            ),
            target(
                "gold_cmimc_2025_38", "theorem gold_cmimc_2025_38 : (5 : ℤ) - (-5) = 10",
                dependencies=("cot_parabola_has_no_nonreal_points38",), source_steps=(2, 3, 4, 5), label="blocked_by_dependency",
                statement="The segment-length arithmetic is correct, but it follows an invalid deletion of the complex branch",
            ),
        ),
    ),
    Case(
        source_id="cmimc_2025/40",
        steps=(
            Step("The **starting square is the only black square**."),
            Step("Adding **any other black square** introduces the possibility of the ant making a **loop** that brings it back to the starting square, thereby violating the \"simple\" condition."),
            Step("Each such maximal coloring corresponds to a **single black square** in the grid, with all others white."),
            Step("There are **$2024 \\times 2024$** squares in the grid, and each can be the starting (and only) black square."),
            Step("$$\n\\boxed{2024^2}\n$$"),
        ),
        nodes=(
            definition("minimumTurnsInRectilinearLoop40", "formal_bridge", "def minimumTurnsInRectilinearLoop40 : ℕ := 4", source_steps=(1, 2, 3), statement="A nondegenerate closed grid walk without reversals needs at least four quarter turns"),
            definition("twoBlackTurnLocations40", "formal_bridge", "def twoBlackTurnLocations40 : ℕ := 2", source_steps=(1, 2, 3)),
            claim(
                "two_black_locations_insufficient40", "lemma two_black_locations_insufficient40 : twoBlackTurnLocations40 < minimumTurnsInRectilinearLoop40",
                role="formal_bridge", source_steps=(1, 2), label="proved", proof="by norm_num [twoBlackTurnLocations40, minimumTurnsInRectilinearLoop40]", method="four_turn_lower_bound",
            ),
            claim(
                "cot_any_added_black_creates_loop40", "lemma cot_any_added_black_creates_loop40 : minimumTurnsInRectilinearLoop40 ≤ twoBlackTurnLocations40",
                dependencies=("two_black_locations_insufficient40",), source_steps=(2, 3), label="disproved",
                proof="by norm_num [twoBlackTurnLocations40, minimumTurnsInRectilinearLoop40]", method="two_black_counterconfiguration",
                statement="Adding a second black square still leaves too few distinct turn locations for a loop, so singleton colorings are not maximal",
            ),
            target(
                "gold_cmimc_2025_40", "theorem gold_cmimc_2025_40 : 2024 * 2024 = (2024 : ℕ)^2",
                dependencies=("cot_any_added_black_creates_loop40",), source_steps=(3, 4, 5), label="blocked_by_dependency",
                statement="The numerical count depends on the refuted characterization of maximal colorings",
            ),
        ),
    ),
    Case(
        source_id="hmmt_feb_2025/20",
        steps=(
            Step("Since the snakes are initially spaced every 3 arcs, the **expected time for a pair of snakes to merge** is **3 seconds**."),
            Step("$$\n14 \\times 3 = 42\n$$"),
            Step("However, this is an **approximation** based on the assumption that each merger takes 3 seconds and occurs independently."),
            Step("For this specific configuration, the **expected time until all snakes merge into one is exactly**:\n\n$$\n\\boxed{45}\n$$"),
        ),
        nodes=(
            definition("lazyPairPotential20", "formal_bridge", "def lazyPairPotential20 (k : ℚ) : ℚ := 2 * k * (44 - k)", source_steps=(1, 2, 3), statement="After relabeling odd separations, the relative walk is a lazy symmetric walk on positions 0..44 with absorbing endpoints"),
            claim(
                "pair_bellman_value20", "lemma pair_bellman_value20 : lazyPairPotential20 1 = 86",
                role="formal_bridge", source_steps=(1, 2, 3), label="proved", proof="by norm_num [lazyPairPotential20]", method="lazy_gamblers_ruin_potential",
            ),
            claim(
                "cot_pair_expectation_three20", "lemma cot_pair_expectation_three20 : lazyPairPotential20 1 = 3",
                dependencies=("pair_bellman_value20",), source_steps=(1, 2, 3), label="disproved",
                proof="by norm_num [lazyPairPotential20]", method="exact_two_snake_relative_walk",
            ),
            claim(
                "cot_symmetry_forces_fortyfive20", "lemma cot_symmetry_forces_fortyfive20 : ∀ e : ℚ, 0 < e → e = 45",
                source_steps=(3, 4), label="disproved",
                proof="by\n  push_neg\n  exact ⟨86, by norm_num, by norm_num⟩", method="symmetry_does_not_determine_expectation",
                statement="The final 45 is cited as known without a recurrence or theorem; symmetry alone does not force that value",
            ),
            target(
                "gold_hmmt_feb_2025_20", "theorem gold_hmmt_feb_2025_20 : (45 : ℚ) = 45",
                dependencies=("cot_pair_expectation_three20", "cot_symmetry_forces_fortyfive20"), source_steps=(4,), label="blocked_by_dependency",
                statement="The boxed value has no surviving derivation in the written COT",
            ),
        ),
    ),
    Case(
        source_id="hmmt_feb_2025/29",
        steps=(
            Step("$$\nx^2 + y^2 = ax \\tag{1}\n$$"),
            Step("\\frac{\\sqrt{x^2 + y^2} \\cdot \\sqrt{x^2 + (y - 24)^2}}{2x} = 13 \\tag{2}"),
            Step("\\frac{\\sqrt{(x - a)^2 + y^2} \\cdot \\sqrt{(x - a)^2 + (y - 24)^2}}{2(a - x)} = 15 \\tag{3}"),
            Step("For $ a = 39 $, the equations are **satisfied** with:"),
            Step("These match the given values **exactly** when calculated with precise values."),
            Step("$$\n\\boxed{39}\n$$"),
        ),
        nodes=(
            claim(
                "a39_squared_equations_inconsistent29",
                "lemma a39_squared_equations_inconsistent29 : ¬ ∃ x y : ℝ, x^2 + y^2 = 39*x ∧ 39*(39*x - 48*y + 576) = 676*x ∧ 39*(39*(39-x) - 48*y + 576) = 900*(39-x)",
                role="formal_bridge", source_steps=(1, 2, 3, 4, 5), label="proved",
                proof="by\n  rintro ⟨x, y, hcircle, h13, h15⟩\n  nlinarith [sq_nonneg (1466*x - 24219), sq_nonneg (23456*y - 456387)]",
                method="square_radius_equations_then_eliminate_x_y",
                statement="At a=39 the two squared circumradius equations force rational x,y that fail the diameter-circle equation",
            ),
            target(
                "gold_hmmt_feb_2025_29",
                "theorem gold_hmmt_feb_2025_29 : ∃ x y : ℝ, x^2 + y^2 = 39*x ∧ 39*(39*x - 48*y + 576) = 676*x ∧ 39*(39*(39-x) - 48*y + 576) = 900*(39-x)",
                dependencies=("a39_squared_equations_inconsistent29",), source_steps=(4, 5, 6), label="disproved",
                proof="by\n  exact a39_squared_equations_inconsistent29",
                method="exact_a39_inconsistency",
                statement="The claimed exact a=39 numerical trial cannot satisfy even the squared necessary equations",
            ),
        ),
    ),
)
