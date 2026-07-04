/-
  Goedel-Architect diagnostic commands built on top of LeanArchitect.
-/
import Lean
import Architect.Basic

open Lean Elab Command Architect

/-- List all @[blueprint]-annotated declarations in the current environment. -/
elab "#list_blueprint" : command => do
  let env ← getEnv
  let nodeMap := (blueprintExt.getState env).get
  let names : List String := nodeMap.foldl (fun acc n _ => acc ++ [n.toString]) []
  if names.isEmpty then
    logInfo "[GoedelArch] No @[blueprint] declarations found."
  else
    logInfo s!"[GoedelArch] Blueprint nodes ({names.length}): {", ".intercalate names}"
