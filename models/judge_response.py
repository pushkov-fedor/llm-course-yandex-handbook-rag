from pydantic import BaseModel


class AnswerJudgeResponse(BaseModel):
    score: int
    explanation: str

    @staticmethod
    def get_response_format() -> dict:
        return {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer_judge_response",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "score": {"type": "integer", "minimum": 1, "maximum": 5},
                                "explanation": {"type": "string"}
                            },
                            "required": ["score", "explanation"],
                            "additionalProperties": False
                        }
                    }
                }

class RefusalJudgeResponse(BaseModel):
    refused: bool
    explanation: str

    @staticmethod
    def get_response_format() -> dict:
        return {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "refusal_judgment",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "refused": {"type": "boolean"},
                                "explanation": {"type": "string"}
                            },
                            "required": ["refused", "explanation"],
                            "additionalProperties": False
                        }
                    }
                }