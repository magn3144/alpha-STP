import REPL.Frontend
import REPL.JSON
import REPL.PaperProof.BetterParser

open Lean Elab

-- Stop at complete source tactics: their macro-generated internals are not
-- additional policy decisions, and can contain enormous shared info trees.
partial def sourceTactics (tree : InfoTree) (outer : Option ContextInfo := none) :
    IO (Array (ContextInfo × TacticInfo)) := do
  match tree with
  | .context ctx child => sourceTactics child (ctx.mergeIntoOuter? outer)
  | .node info children =>
    if let .ofTacticInfo tactic := info then
      if info.isOriginal && tactic.isSubstantive then
        let some ctx := outer | throw (IO.userError "Missing tactic context")
        return #[(ctx, tactic)]
    let mut found := #[]
    for child in children.toArray do
      found := found ++ (← sourceTactics child outer)
    return found
  | .hole _ => return #[]

-- Collapse internal metavariables onto the next recorded source goals.
partial def nextSourceGoals (mctx : MetavarContext) (known : Std.HashSet MVarId)
    (goal : MVarId) : StateM (Std.HashSet MVarId) (Array MVarId) := do
  if known.contains goal then return #[goal]
  if (← get).contains goal then return #[]
  modify (·.insert goal)
  let some expression := mctx.eAssignment.find? goal | return #[]
  let mut found := #[]
  for child in (expression.collectMVars {}).result do
    found := found ++ (← nextSourceGoals mctx known child)
  return found

structure SourceStep where
  context : ContextInfo
  tactic : TacticInfo
  before : List MVarId

unsafe def traceSource (source : String) (initial : Command.State) : IO (Array Json) := do
  let body := String.intercalate "\n" (source.splitOn "\n" |>.filter (!·.startsWith "import "))
  let (_, _, messages, trees) ← Lean.Elab.IO.processInput body (some initial)
  for message in messages do
    if message.severity == .error then
      throw (IO.userError (← message.data.toString))
  let mut steps : Array SourceStep := #[]
  let mut known : Std.HashSet MVarId := {}
  for tree in trees do
    for (context, tactic) in (← sourceTactics tree) do
      let before ← PaperProof.getUnassignedGoals tactic.goalsBefore tactic.mctxBefore
      let changed := before.filter fun goal =>
        !tactic.goalsAfter.contains goal || tactic.mctxAfter.eAssignment.contains goal ||
          tactic.mctxAfter.dAssignment.contains goal
      steps := steps.push ⟨context, tactic, changed⟩
      known := known.insertMany changed
  let mut records := #[]
  for step in steps do
    let some source := step.tactic.stx.getSubstring?
      | throw (IO.userError "Missing original tactic source")
    for before in step.before do
      let mut candidates : Array MVarId := #[]
      if let some expression := step.tactic.mctxAfter.eAssignment.find? before then
        candidates := (expression.collectMVars {}).result
      if step.before.length == 1 then
        candidates := candidates ++ (step.tactic.goalsAfter.filter
          (!step.tactic.goalsBefore.contains ·)).toArray
      let mut after : Std.HashSet MVarId := {}
      for candidate in candidates do
        let (found, _) := (nextSourceGoals step.tactic.mctxAfter known candidate).run {}
        after := after.insertMany found
      let goal ← REPL.printGoalInfo {step.context with mctx := step.tactic.mctxBefore} before
      records := records.push (Json.mkObj [
        ("tacticString", toJson source.toString),
        ("goalBefore", toJson goal),
        ("children", toJson (after.toArray.map (·.name.toString)))])
  return records

structure InputProof where
  index : Nat
  path : String
deriving FromJson

unsafe def main (args : List String) : IO Unit := do
  let request ← IO.ofExcept (Json.parse (← IO.FS.readFile args.head!))
  let proofs ← IO.ofExcept (fromJson? (α := Array InputProof) request)
  let (_, initial, _, _) ← Lean.Elab.IO.processInput "import Mathlib\nset_option maxHeartbeats 0" none
  for proof in proofs do
    let records ← traceSource (← IO.FS.readFile proof.path) initial
    IO.println (Json.mkObj [("index", toJson proof.index), ("steps", toJson records)]).compress
    (← IO.getStdout).flush
