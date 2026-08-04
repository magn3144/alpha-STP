"""Calculate both difficulty metrics on the problems in the given dataset.
These difficulty metrics are the STP and AlphaProof metrics.
The point is to see how they correlate"""

from stp.core.config import load_config
from stp.evaluate_difficulty_score.data import (
    alphaproof_attempts_by_problem,
    evaluation_paths,
    llm_attempts_by_problem,
    load_problems,
    load_results_by_problem,
    requests_by_problem,
    save_alphaproof_search,
    save_llm_generations,
    statement_ids,
)
from stp.evaluate_difficulty_score.evaluation_config import (
    DATASET_PATH,
    LLM_ATTEMPTS,
    alphaproof_solver_config,
    evaluation_directory,
    llm_solver_config,
    model_paths,
    parse_args,
    select_prover_handler,
)
from stp.evaluate_difficulty_score.scores import (
    append_score,
    recover_missing_scores,
    score_keys,
)
from stp.evaluate_difficulty_score.solver_processes import (
    solve_with_alphaproof_process,
    solve_with_llm_process,
)
from stp.self_play.generate import make_proof_requests


def evaluate() -> None:
    """Run the complete resumable Numina evaluation from CLI inputs."""

    args = parse_args()
    config = load_config(args.config)
    config = select_prover_handler(config, args.llm_prover_handler)
    output_dir = evaluation_directory(args.name)
    model, tokenizer = model_paths(args, config)

    problems = load_problems(DATASET_PATH)
    problem_ids = statement_ids(problems)
    paths = evaluation_paths(output_dir)

    llm_results = llm_attempts_by_problem(
        load_results_by_problem(paths.llm, problem_ids)
    )
    alphaproof_results = alphaproof_attempts_by_problem(
        load_results_by_problem(paths.alphaproof, problem_ids)
    )
    saved_scores = score_keys(paths.scores, problem_ids)

    recover_missing_scores(
        paths.scores,
        llm_results,
        alphaproof_results,
        saved_scores,
    )

    llm_config = llm_solver_config(config)
    alphaproof_config = alphaproof_solver_config(config)
    llm_requests = make_proof_requests(problems, LLM_ATTEMPTS, config.run.seed)
    alphaproof_requests = make_proof_requests(problems, 1, config.run.seed)
    llm_requests_by_id = requests_by_problem(llm_requests)
    alphaproof_requests_by_id = requests_by_problem(alphaproof_requests)

    for problem in problems:
        if problem.id not in llm_results:
            llm_attempts = solve_with_llm_process(
                llm_requests_by_id[problem.id],
                llm_config,
                model,
                tokenizer,
            )
            save_llm_generations(
                paths.llm,
                problem,
                llm_attempts,
                llm_results,
            )
            append_score(paths.scores, llm_attempts, saved_scores)

        if problem.id not in alphaproof_results:
            alphaproof_attempts, raw_result = solve_with_alphaproof_process(
                alphaproof_requests_by_id[problem.id],
                alphaproof_config,
            )
            save_alphaproof_search(
                paths.alphaproof,
                problem,
                alphaproof_attempts,
                raw_result,
                alphaproof_results,
            )
            append_score(paths.scores, alphaproof_attempts, saved_scores)

    print(output_dir)


if __name__ == "__main__":
    evaluate()
