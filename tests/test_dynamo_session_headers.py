from ThunderAgent.app import backend_context_headers, get_program_id


def test_dynamo_session_header_drives_program_id_and_backend_context():
    headers = {
        "x-dynamo-session-id": "root-session",
        "x-dynamo-parent-session-id": "parent-session",
        "authorization": "Bearer ignored",
    }

    assert get_program_id({}, headers) == "root-session"
    assert backend_context_headers(headers) == {
        "x-dynamo-session-id": "root-session",
        "x-dynamo-parent-session-id": "parent-session",
    }


def test_nvext_agent_context_session_id_fallback():
    assert (
        get_program_id({"nvext": {"agent_context": {"session_id": "nvext-session"}}})
        == "nvext-session"
    )
