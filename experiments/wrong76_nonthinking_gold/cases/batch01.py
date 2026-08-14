from __future__ import annotations

from gold_dsl import Case, Step, claim, definition, target


CASES = (
    Case(
        source_id="MATH-500/test/prealgebra/1139.json",
        steps=(
            Step("The expression evaluates as:\n\n$$\n(2 \\cdot 3 \\cdot 4 \\cdot 5) + 1 = 120 + 1 = 121\n$$"),
            Step("The key is that the **only addition** in the expression is the one at the end: `+1`. We can insert parentheses to **change when** that addition is performed, and also **group multiplications** in different ways."),
            Step("Some of the resulting values we can obtain are:"),
            Step("These are all **distinct** values that can be obtained by valid parenthesization of the original expression."),
            Step("From the above list, we have **12 distinct values** that can be obtained by inserting parentheses in the expression without rearranging terms."),
        ),
        nodes=(
            definition(
                "Paren5", "problem_grounding",
                "abbrev Paren5 := Fin 14",
                problem_source_span="only inserting parentheses; operations and term order are fixed",
                statement="The fourteen full binary parenthesizations of five ordered operands",
            ),
            definition(
                "parenValue", "problem_grounding",
                "def parenValue (p : Paren5) : ℕ := match p.val with\n"
                "  | 0 => 2 * (3 * (4 * (5 + 1)))\n"
                "  | 1 => 2 * (3 * ((4 * 5) + 1))\n"
                "  | 2 => 2 * ((3 * 4) * (5 + 1))\n"
                "  | 3 => 2 * ((3 * (4 * 5)) + 1)\n"
                "  | 4 => 2 * (((3 * 4) * 5) + 1)\n"
                "  | 5 => (2 * 3) * (4 * (5 + 1))\n"
                "  | 6 => (2 * 3) * ((4 * 5) + 1)\n"
                "  | 7 => (2 * (3 * 4)) * (5 + 1)\n"
                "  | 8 => ((2 * 3) * 4) * (5 + 1)\n"
                "  | 9 => (2 * (3 * (4 * 5))) + 1\n"
                "  | 10 => (2 * ((3 * 4) * 5)) + 1\n"
                "  | 11 => ((2 * 3) * (4 * 5)) + 1\n"
                "  | 12 => ((2 * (3 * 4)) * 5) + 1\n"
                "  | _ => (((2 * 3) * 4) * 5) + 1",
                problem_source_span="2·3·4·5+1 with no rearrangement or operator changes",
                statement="Evaluate each legal parenthesization without changing tokens",
            ),
            definition(
                "obtainableValues", "formal_bridge",
                "def obtainableValues : Finset ℕ := Finset.univ.image parenValue",
                source_steps=(2, 3), statement="The values of all legal parenthesizations",
            ),
            claim(
                "original_expression_value", "lemma original_expression_value : parenValue ⟨13, by decide⟩ = 121",
                source_steps=(1,), label="proved", proof="by decide", method="finite_reduction",
                statement="The ordinary parenthesization evaluates to 121",
            ),
            claim(
                "cot_claims_160_obtainable", "lemma cot_claims_160_obtainable : ∃ p, parenValue p = 160",
                source_steps=(3,), label="disproved", proof="by decide", method="enumerate_fourteen_parenthesizations",
                statement="The COT lists 160 as obtainable by parentheses alone",
            ),
            target(
                "gold_preallgebra_1139", "theorem gold_preallgebra_1139 : obtainableValues.card = 12",
                dependencies=("original_expression_value", "cot_claims_160_obtainable"), source_steps=(4, 5),
                label="blocked_by_dependency", statement="The COT concludes that twelve distinct values are obtainable",
            ),
        ),
        fidelity_notes=(
            "The grammar preserves all four original operators; the COT's expression for 160 changes an operator and is refuted by exhaustive evaluation.",
        ),
    ),
    Case(
        source_id="MATH-500/test/geometry/434.json",
        steps=(
            Step("The angle between **AB** and the **top horizontal line** at point **A** is given as **124°**."),
            Step("Therefore, the angle between **AB** and **BC** at point **B** is:\n  \n  $$\n  180^\\circ - 124^\\circ = 56^\\circ\n  $$"),
            Step("Using the triangle angle sum:"),
            Step("So, the **angle at point A in triangle ABC** is **62°**."),
            Step("Therefore, **x° = 62°**."),
        ),
        nodes=(
            definition(
                "diagramAngleRelations434", "problem_grounding",
                "def diagramAngleRelations434 (x q : ℝ) : Prop := 2 * x = 128 ∧ q + x = 180",
                problem_source_span="PT is parallel to QR; the marked 2x and 128 angles are alternate interior angles, while x and angle PQR are same-side interior angles",
                statement="Correct angle relations determined by the marked regions and parallel lines",
            ),
            claim(
                "cot_supplementary_relation434",
                "lemma cot_supplementary_relation434 (x q : ℝ) (h : diagramAngleRelations434 x q) : 2 * x + 128 = 180",
                source_steps=(1, 2), label="disproved",
                proof="by\n  push_neg\n  refine ⟨64, 116, ?_, ?_⟩\n  · norm_num [diagramAngleRelations434]\n  · norm_num",
                method="marked_angle_witness_64_116",
                statement="The COT treats the marked 2x angle and the exterior 128 angle as supplementary",
            ),
            claim(
                "cot_triangle_base_angle434",
                "lemma cot_triangle_base_angle434 (x q : ℝ) (h : diagramAngleRelations434 x q) : q = 62",
                dependencies=("cot_supplementary_relation434",), source_steps=(3, 4),
                label="blocked_by_dependency", statement="The COT derives a 62 degree target angle",
            ),
            target(
                "gold_geometry_434", "theorem gold_geometry_434 (x q : ℝ) (h : diagramAngleRelations434 x q) : q = 62",
                dependencies=("cot_triangle_base_angle434",), source_steps=(5,),
                label="blocked_by_dependency", statement="The COT final answer is 62 degrees",
            ),
        ),
        fidelity_notes=("The diagram coordinates are used only to identify the marked angle regions, as required by the agreed source policy.",),
    ),
    Case(
        source_id="MATH-500/test/geometry/880.json",
        steps=(
            Step("The gecko is on the **left wall** (y = 0), **one foot from the ceiling** (z = 7), and **one foot from the back wall** (x = 1), so its position is:\n\n$$\n(1, 0, 7)\n$$"),
            Step("The fly is on the **right wall** (y = 10), **one foot from the floor** (z = 1), and **one foot from the front wall** (x = 11), so its position is:\n\n$$\n(11, 10, 1)\n$$"),
            Step("And the fly is at:\n\n$$\n(11, 10 + 1) = (11, 11)\n$$"),
            Step("Now, we compute the straight-line distance between the two points in this unfolded plane:"),
            Step("$$\n\\boxed{2\\sqrt{29}}\n$$"),
        ),
        nodes=(
            definition("roomHeight880", "problem_grounding", "def roomHeight880 : ℕ := 8", problem_source_span="8 feet tall"),
            definition("roomWidth880", "problem_grounding", "def roomWidth880 : ℕ := 10", problem_source_span="10 feet wide"),
            definition("geckoHeight880", "cot_claim", "def geckoHeight880 : ℕ := 7", source_steps=(1,)),
            definition("flyHeight880", "cot_claim", "def flyHeight880 : ℕ := 1", source_steps=(2,)),
            definition(
                "correctCeilingOffset880", "formal_bridge",
                "def correctCeilingOffset880 : ℕ := (roomHeight880 - geckoHeight880) + roomWidth880 + (roomHeight880 - flyHeight880)",
                source_steps=(3,), statement="Surface separation after unfolding both side walls away from the ceiling",
            ),
            definition(
                "cotCeilingOffset880", "cot_claim", "def cotCeilingOffset880 : ℕ := (roomWidth880 + flyHeight880) - geckoHeight880",
                source_steps=(3,), statement="The COT's unfolded vertical separation",
            ),
            claim(
                "correct_ceiling_offset880", "lemma correct_ceiling_offset880 : correctCeilingOffset880 = 18",
                role="formal_bridge", source_steps=(3,), label="proved", proof="by norm_num [correctCeilingOffset880, roomHeight880, roomWidth880, geckoHeight880, flyHeight880]",
                method="unfold_surface_offsets", statement="The two wall-to-ceiling offsets plus ceiling width total 18",
            ),
            claim(
                "cot_unfolding_preserves_distance880", "lemma cot_unfolding_preserves_distance880 : cotCeilingOffset880 = correctCeilingOffset880",
                dependencies=("correct_ceiling_offset880",), source_steps=(3,), label="disproved",
                proof="by norm_num [cotCeilingOffset880, correctCeilingOffset880, roomHeight880, roomWidth880, geckoHeight880, flyHeight880]",
                method="compare_unfolded_offsets_4_and_18", statement="The COT's point (11,11) is asserted to be the correct unfolding",
            ),
            claim(
                "cot_distance_simplification880", "lemma cot_distance_simplification880 : Real.sqrt 116 = 2 * Real.sqrt 29",
                dependencies=("cot_unfolding_preserves_distance880",), source_steps=(4,),
                label="blocked_by_dependency", statement="The COT simplifies its candidate distance",
            ),
            target(
                "gold_geometry_880", "theorem gold_geometry_880 : Real.sqrt 116 = 2 * Real.sqrt 29",
                dependencies=("cot_distance_simplification880",), source_steps=(5,),
                label="blocked_by_dependency", statement="The COT claims the shortest path is 2 sqrt 29",
            ),
        ),
    ),
    Case(
        source_id="MATH-500/test/prealgebra/1003.json",
        steps=(
            Step("The **heptagon** is a regular polygon with 7 sides, and each internal angle is:\n  $$\n  \\text{Internal angle} = \\frac{(7-2) \\cdot 180^\\circ}{7} = \\frac{900^\\circ}{7}\n  $$"),
            Step("Thus, the angle $ \\angle BAC $ is the difference between these two angles:"),
            Step("Using vector arguments and the properties of the heptagon and square, we find that the angle between $ \\vec{AB} $ and $ \\vec{AC} $, where $ \\vec{AB} $ is the side of the square adjacent to $ A $ and $ \\vec{AC} $ is obtained by rotating $ \\vec{AF} $ by $ 90^\\circ $, is:"),
            Step("Therefore, the measure of $ \\angle BAC $ is:\n\n$$\n\\boxed{\\dfrac{990}{7}}\n$$"),
        ),
        nodes=(
            definition("heptagonInterior1003", "cot_claim", "def heptagonInterior1003 : ℚ := 900 / 7", source_steps=(1,)),
            claim(
                "heptagon_interior_formula1003",
                "lemma heptagon_interior_formula1003 : (((7 - 2) * 180 : ℚ) / 7) = heptagonInterior1003",
                source_steps=(1,), label="proved", proof="by norm_num [heptagonInterior1003]", method="rational_normalization",
            ),
            claim(
                "cot_revised_vector_angle1003",
                "lemma cot_revised_vector_angle1003 : (90 : ℚ) + (heptagonInterior1003 - 90) = heptagonInterior1003",
                dependencies=("heptagon_interior_formula1003",), source_steps=(2, 3),
                label="proved", proof="by ring", method="ring",
            ),
            target(
                "gold_preallgebra_1003",
                "theorem gold_preallgebra_1003 : heptagonInterior1003 = (990 : ℚ) / 7",
                dependencies=("cot_revised_vector_angle1003",), source_steps=(4,), label="disproved",
                proof="by norm_num [heptagonInterior1003]", method="final_numerator_mismatch_900_990",
                statement="The final boxed numerator changes from 900 to 990",
            ),
        ),
    ),
    Case(
        source_id="MATH-500/test/prealgebra/378.json",
        steps=(
            Step("The triangle is formed by the points `(0, 4)`, `(0, 0)`, and `(2, 0)`."),
            Step("The triangle is formed by the points `(0, 4)`, `(3, 4)`, and `(3, 0)`."),
            Step("The first triangle lies entirely within the second triangle."),
            Step("Since the line from `(0, 4)` to `(2, 0)` is **part** of the line from `(0, 4)` to `(3, 0)`, the first triangle lies **entirely within** the second triangle."),
            Step("$$\n\\boxed{6}\n$$"),
        ),
        nodes=(
            definition("steepLine378", "cot_claim", "def steepLine378 (x : ℚ) : ℚ := -2 * x + 4", source_steps=(1, 4)),
            definition("wideLine378", "cot_claim", "def wideLine378 (x : ℚ) : ℚ := -(4 / 3) * x + 4", source_steps=(2, 4)),
            claim(
                "first_triangle_area378", "lemma first_triangle_area378 : (1 : ℚ) / 2 * 2 * 4 = 4",
                source_steps=(1,), label="proved", proof="by norm_num", method="norm_num",
            ),
            claim(
                "second_triangle_area378", "lemma second_triangle_area378 : (1 : ℚ) / 2 * 3 * 4 = 6",
                source_steps=(2,), label="proved", proof="by norm_num", method="norm_num",
            ),
            claim(
                "cot_lines_are_same378", "lemma cot_lines_are_same378 : ∀ x : ℚ, steepLine378 x = wideLine378 x",
                source_steps=(3, 4), label="disproved",
                proof="by\n  push_neg\n  exact ⟨1, by norm_num [steepLine378, wideLine378]⟩",
                method="x_one_distinguishes_slopes", statement="The COT says the two different-slope boundaries are parts of the same line",
            ),
            target(
                "gold_preallgebra_378",
                "theorem gold_preallgebra_378 : (∀ x : ℚ, steepLine378 x = wideLine378 x) ∧ (1 : ℚ) / 2 * 3 * 4 = 6",
                dependencies=("first_triangle_area378", "cot_lines_are_same378", "second_triangle_area378"), source_steps=(5,),
                label="blocked_by_dependency", statement="The COT concludes the union has area six from the containment claim",
            ),
        ),
    ),
    Case(
        source_id="MATH-500/test/counting_and_probability/765.json",
        steps=(
            Step("Therefore, students taking **at least one** subject:  \n  $$\n  360 - 15 = 345\n  $$"),
            Step("Therefore, the total number taking both **physics and calculus** (including those also taking chemistry) is:\n  $$\n  |C \\cap P| = 30 + 15 = 45\n  $$"),
            Step("Substitute known values:\n\n$$\n345 = 180 + P + 2P - 45 - 75 - 75 + 15\n$$"),
            Step("$$\n3P = 345 \\Rightarrow P = 115\n$$"),
            Step("This matches the number of students taking **at least one** subject, confirming our calculation is correct."),
        ),
        nodes=(
            definition("atLeastOne765", "cot_claim", "def atLeastOne765 : ℤ := 360 - 15", source_steps=(1,)),
            definition("problemCalcPhysTotal765", "problem_grounding", "def problemCalcPhysTotal765 : ℤ := 30", problem_source_span="Only 30 take both physics and calculus"),
            claim(
                "at_least_one_value765", "lemma at_least_one_value765 : atLeastOne765 = 345",
                source_steps=(1,), label="proved", proof="by norm_num [atLeastOne765]", method="norm_num",
            ),
            claim(
                "correct_inclusion_exclusion765",
                "lemma correct_inclusion_exclusion765 : atLeastOne765 = 180 + (110 : ℤ) + 2 * 110 - problemCalcPhysTotal765 - 75 - 75 + 15",
                dependencies=("at_least_one_value765",), source_steps=(3,),
                label="proved", proof="by norm_num [atLeastOne765, problemCalcPhysTotal765]", method="inclusion_exclusion_with_total_CP_thirty",
            ),
            target(
                "gold_counting_765",
                "theorem gold_counting_765 : (110 : ℤ) = 115",
                dependencies=("correct_inclusion_exclusion765",), source_steps=(2, 4, 5),
                label="disproved", proof="by norm_num", method="correct_inclusion_exclusion_gives_110",
                statement="The problem's 30 is already the total C∩P count; using it gives 110 physics students, not 115",
            ),
        ),
    ),
    Case(
        source_id="MATH-500/test/precalculus/1056.json",
        steps=(
            Step("$$\n\\frac{(x - z)^2}{2} + \\frac{(x - 2y + z)^2}{6} + \\frac{(x + y + z)^2}{3} = 36\n$$"),
            Step("So the entire expression simplifies to:\n\n$$\n6x^2 + 6y^2 + 6z^2 = 216\n$$"),
            Step("$$\nx^2 + y^2 + z^2 = 36\n$$"),
            Step("The **region of the volume enclosed by $ S $** is then the **interior** of this sphere, i.e., the set of all points satisfying:\n\n$$\nx^2 + y^2 + z^2 \\leq 36\n$$"),
            Step("$$\n\\boxed{x^2 + y^2 + z^2 = 36}\n$$"),
        ),
        nodes=(
            definition("Point3_1056", "problem_grounding", "abbrev Point3_1056 := Fin 3 → ℝ", problem_source_span="P=(x,y,z)"),
            definition(
                "distanceSquareSum1056", "cot_claim",
                "noncomputable def distanceSquareSum1056 (x y z : ℝ) : ℝ := (x - z)^2 / 2 + (x - 2*y + z)^2 / 6 + (x + y + z)^2 / 3",
                source_steps=(1,),
            ),
            definition("normSquare1056", "formal_bridge", "def normSquare1056 (p : Point3_1056) : ℝ := p 0 ^ 2 + p 1 ^ 2 + p 2 ^ 2", source_steps=(2, 3)),
            definition("sphereSurface1056", "cot_claim", "def sphereSurface1056 : Set Point3_1056 := {p | normSquare1056 p = 36}", source_steps=(3, 5)),
            definition("enclosedRegion1056", "cot_claim", "def enclosedRegion1056 : Set Point3_1056 := {p | normSquare1056 p ≤ 36}", source_steps=(4,)),
            claim(
                "distance_sum_identity1056",
                "lemma distance_sum_identity1056 (x y z : ℝ) : distanceSquareSum1056 x y z = x^2 + y^2 + z^2",
                source_steps=(1, 2, 3), label="proved", proof="by\n  unfold distanceSquareSum1056\n  ring",
                method="field_simp_ring", statement="The three squared plane distances sum to x squared plus y squared plus z squared",
            ),
            target(
                "gold_precalculus_1056", "theorem gold_precalculus_1056 : enclosedRegion1056 = sphereSurface1056",
                dependencies=("distance_sum_identity1056",), source_steps=(4, 5), label="disproved",
                proof="by\n  intro h\n  let o : Point3_1056 := fun _ => 0\n  have ho : o ∈ enclosedRegion1056 := by norm_num [o, enclosedRegion1056, normSquare1056]\n  rw [h] at ho\n  norm_num [o, sphereSurface1056, normSquare1056] at ho",
                method="origin_is_inside_not_on_surface", statement="The final boxed surface equation is asserted to be the enclosed region",
            ),
        ),
    ),
    Case(
        source_id="MATH-500/test/prealgebra/874.json",
        steps=(
            Step("Since $ PT \\parallel QR $, and $ TR $ is a transversal intersecting both, the angles at $ T $ and $ R $ are **same-side interior angles**. Therefore:"),
            Step("$$\n2x + 128 = 180\n$$"),
            Step("$$\n2x = 52 \\Rightarrow x = 26\n$$"),
            Step("Thus:\n\n$$\n\\angle PQR = x = 26^\\circ\n$$"),
            Step("$$\n\\boxed{26}\n$$"),
        ),
        nodes=(
            definition(
                "diagramAngleRelations874", "problem_grounding",
                "def diagramAngleRelations874 (x q : ℝ) : Prop := 2 * x = 128 ∧ q + x = 180",
                problem_source_span="PT is parallel to QR with x, 2x, and exterior 128 degree angles marked in the shown regions",
                statement="Correct alternate-interior and same-side-interior relations from the marked regions",
            ),
            claim(
                "cot_same_side_relation874",
                "lemma cot_same_side_relation874 (x q : ℝ) (h : diagramAngleRelations874 x q) : 2 * x + 128 = 180",
                source_steps=(1, 2), label="disproved",
                proof="by\n  push_neg\n  refine ⟨64, 116, ?_, ?_⟩\n  · norm_num [diagramAngleRelations874]\n  · norm_num",
                method="marked_angle_witness_64_116", statement="The COT treats alternate interior marked angles as supplementary",
            ),
            claim(
                "cot_corresponding_angle874",
                "lemma cot_corresponding_angle874 (x q : ℝ) (h : diagramAngleRelations874 x q) : q = x",
                source_steps=(4,), label="disproved",
                proof="by\n  push_neg\n  refine ⟨64, 116, ?_, ?_⟩\n  · norm_num [diagramAngleRelations874]\n  · norm_num",
                method="same_side_angles_are_supplementary_not_equal", statement="The COT equates the same-side interior angles at P and Q",
            ),
            claim(
                "cot_solves_x874", "lemma cot_solves_x874 (x q : ℝ) (h : diagramAngleRelations874 x q) : x = 26",
                dependencies=("cot_same_side_relation874",), source_steps=(3,), label="blocked_by_dependency",
            ),
            target(
                "gold_preallgebra_874", "theorem gold_preallgebra_874 (x q : ℝ) (h : diagramAngleRelations874 x q) : q = 26",
                dependencies=("cot_corresponding_angle874", "cot_solves_x874"), source_steps=(5,),
                label="blocked_by_dependency", statement="The COT final answer is 26 degrees",
            ),
        ),
    ),
)
