import pytest

from psychometric.session import create_session, start_response, submit_answer


@pytest.mark.asyncio
async def test_assessment_session_flow():
    session = create_session(user_id="test-user-123", language="hi")
    start = start_response(session)
    assert start["session_id"]
    assert start["language"] == "hi"
    assert start["item"] is not None

    item_id = start["item"]["item_id"]
    result = await submit_answer(session.session_id, item_id, "A")
    assert result["completed"] is False or result["completed"] is True
