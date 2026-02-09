import asyncio
import logging

import mlflow
from mlflow.entities import Feedback as MLFlowFeedback
from mlflow.tracing.assessment import log_assessment
from scenario_server.entities import ScenarioAnswer, SubmissionResult

logger: logging.Logger = logging.getLogger(__name__)
logger.debug(f"debug: {__name__}")


def mlflow_logging(tracking_context, results: SubmissionResult):
    logger.debug(f"{tracking_context=}")

    experiment_id: str = tracking_context["experiment_id"]
    run_id: str = tracking_context["run_id"]

    mlflow.set_experiment(experiment_id=experiment_id)
    with mlflow.start_run(run_id=run_id):
        traces = mlflow.search_traces(experiment_ids=[experiment_id], run_id=run_id)
        logger.debug(f"{len(traces)=}")

        for grade in results.grades:
            result_id: str = grade.scenario_id

            mask = traces["tags"].apply(
                lambda d: isinstance(d, dict) and d.get("scenario_id") == result_id
            )
            trace_row = traces[mask]

            try:
                tid = trace_row.iloc[0]["trace_id"]
                feedback = MLFlowFeedback(name="Correct", value=grade.correct)
                log_assessment(trace_id=tid, assessment=feedback)

            except Exception as e:
                logger.exception(f"failed to log result: {e=}")

            for r in grade.details:
                try:
                    tid = trace_row.iloc[0]["trace_id"]
                    if isinstance(r, MLFlowFeedback):
                        log_assessment(trace_id=tid, assessment=r)
                    else:
                        log_assessment(
                            trace_id=tid,
                            assessment=MLFlowFeedback(
                                name=r["name"],
                                value=r["value"],
                            ),
                        )
                except Exception as e:
                    logger.exception(f"failed to log assessment: {e=}")

        try:
            for summary in results.summary:
                k: str = summary.name
                v: str = summary.value
                mlflow.set_tag(k, v)
        except Exception as e:
            logger.exception(f"failed to set summary tag")


async def grade_responses(grader, data) -> SubmissionResult:
    submission: list[ScenarioAnswer] = [
        ScenarioAnswer(scenario_id=s["scenario_id"], answer=s["answer"])
        for s in data["submission"]
    ]

    tracking_context = data.get("tracking_context", None)
    results = await grader(submission)

    if tracking_context:
        # mlflow uses blocking sends which can block the event loop
        await asyncio.to_thread(mlflow_logging, tracking_context, results)

    return results
