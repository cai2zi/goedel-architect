/-
  Goedel-Architect validation command built on top of LeanArchitect.

  `#validate_blueprint <target>` checks that the main theorem `<target>` is
  present among the @[blueprint]-annotated declarations (via LeanArchitect's
  blueprintExt) and emits the "Compilation SUCCESSFUL. Validation SUCCESSFUL."
  stamp that the Python pipeline watches for.
-/
import Lean
import Architect.Basic

open Lean Elab Command Architect

elab "#validate_blueprint" target:ident : command => do
  let targetName := target.getId
  let env ← getEnv
  match blueprintExt.find? env targetName with
  | none =>
    throwError
      s!"[Architect] Main theorem '{targetName}' not found among @[blueprint] nodes. \
         Ensure the theorem name matches exactly and carries the @[blueprint] attribute."
  | some _ =>
    logInfo "Compilation SUCCESSFUL. Validation SUCCESSFUL."
