import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from agno.tools.twittrapi import TwittrAPITools

PAGE = {
    "data": [
        {
            "id": "1",
            "url": "https://x.com/agno_agi/status/1",
            "text": "Agno 3 is out",
            "author": {"id": "42", "username": "agno_agi", "name": "Agno"},
            "counts": {"likes": 10},
        },
        {"id": "2", "unavailable": True},
    ],
    "nextCursor": "CURSOR",
    "previousCursor": None,
    "hasNextPage": True,
}
LAST_PAGE = {"data": [], "nextCursor": None, "previousCursor": None, "hasNextPage": False}
USER = {"data": {"id": "42", "username": "agno_agi", "name": "Agno", "counts": {"followers": 1000}}}


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    monkeypatch.delenv("TWITTRAPI_API_KEY", raising=False)


def _response(status_code=200, body=None):
    return httpx.Response(status_code=status_code, json=PAGE if body is None else body)


def _tools(**kwargs):
    return TwittrAPITools(api_key="test_key", **kwargs)


def test_registers_all_functions_by_default():
    tools = _tools()
    assert set(tools.functions) == {
        "search_tweets",
        "get_user",
        "get_user_tweets",
        "get_followers",
        "get_following",
        "get_tweet",
        "get_trends",
    }
    assert set(tools.async_functions) == set(tools.functions)


def test_enable_flags_select_functions():
    tools = _tools(
        enable_get_user=False,
        enable_get_user_tweets=False,
        enable_get_followers=False,
        enable_get_following=False,
        enable_get_tweet=False,
        enable_get_trends=False,
    )
    assert list(tools.functions) == ["search_tweets"]
    assert len(_tools(enable_search_tweets=False, enable_get_user=False, all=True).functions) == 7


def test_api_key_from_env(monkeypatch):
    monkeypatch.setenv("TWITTRAPI_API_KEY", "env_key")
    assert TwittrAPITools().api_key == "env_key"


def test_missing_key_returns_error_without_request():
    tools = TwittrAPITools()
    with patch("agno.tools.twittrapi.httpx.get") as mock_get:
        result = tools.search_tweets("agno")
    mock_get.assert_not_called()
    assert result.startswith("Error: TWITTRAPI_API_KEY is not set")


def test_search_tweets_request_and_output():
    tools = _tools()
    with patch("agno.tools.twittrapi.httpx.get", return_value=_response()) as mock_get:
        result = tools.search_tweets("from:agno_agi lang:en", count=5)

    assert mock_get.call_args.args[0] == "https://api.twittrapi.com/search"
    assert mock_get.call_args.kwargs["params"] == {
        "searchItem": "from:agno_agi lang:en",
        "category": "Latest",
        "count": 5,
        "format": "simple",
    }
    assert mock_get.call_args.kwargs["headers"]["X-API-KEY"] == "test_key"
    header, body = result.split("\n\n", 1)
    assert header.startswith("next_cursor: CURSOR")
    data = json.loads(body)
    # unavailable placeholders are dropped
    assert [t["id"] for t in data["data"]] == ["1"]


def test_search_rejects_empty_query():
    with patch("agno.tools.twittrapi.httpx.get") as mock_get:
        assert _tools().search_tweets("  ") == "Error: query must not be empty."
    mock_get.assert_not_called()


def test_last_page_has_no_cursor():
    with patch("agno.tools.twittrapi.httpx.get", return_value=_response(body=LAST_PAGE)):
        result = _tools().search_tweets("agno", cursor="OLD")
    assert result.startswith("next_cursor: none (last page)")


def test_get_user_strips_at_sign():
    with patch("agno.tools.twittrapi.httpx.get", return_value=_response(body=USER)) as mock_get:
        result = _tools().get_user("@agno_agi")
    assert mock_get.call_args.args[0] == "https://api.twittrapi.com/user/by/username/agno_agi"
    assert json.loads(result)["data"]["username"] == "agno_agi"
    assert not result.startswith("next_cursor")


def test_user_timeline_by_username_uses_server_side_lookup():
    with patch("agno.tools.twittrapi.httpx.get", return_value=_response()) as mock_get:
        _tools().get_user_tweets(username="agno_agi", count=10)
    assert mock_get.call_args.args[0] == "https://api.twittrapi.com/user/-1/tweets"
    assert mock_get.call_args.kwargs["params"] == {"username": "agno_agi", "count": 10, "format": "simple"}


def test_user_id_wins_over_username():
    with patch("agno.tools.twittrapi.httpx.get", return_value=_response()) as mock_get:
        _tools().get_followers(username="ignored", user_id="42")
        _tools().get_following(user_id="42")
    paths = [call.args[0] for call in mock_get.call_args_list]
    assert paths == ["https://api.twittrapi.com/user/42/followers", "https://api.twittrapi.com/user/42/followings"]


def test_user_timeline_needs_a_user():
    assert _tools().get_user_tweets() == "Error: Provide either username or user_id."


def test_get_tweet_validates_id():
    assert _tools().get_tweet("https://x.com/a/status/1") == "Error: tweet_id must be a numeric tweet ID."
    with patch("agno.tools.twittrapi.httpx.get", return_value=_response()) as mock_get:
        _tools().get_tweet("123", cursor="C")
    assert mock_get.call_args.args[0] == "https://api.twittrapi.com/tweet/123"
    assert mock_get.call_args.kwargs["params"] == {"cursor": "C", "format": "simple"}


def test_get_trends_defaults_to_worldwide():
    with patch("agno.tools.twittrapi.httpx.get", return_value=_response(body={"data": []})) as mock_get:
        _tools().get_trends()
    assert mock_get.call_args.args[0] == "https://api.twittrapi.com/trends/1"


@pytest.mark.parametrize(
    "status, hint",
    [(401, "missing or invalid"), (402, "balance is empty"), (429, "rate limited"), (500, "request failed")],
)
def test_http_errors_are_returned_to_the_agent(status, hint):
    with patch("agno.tools.twittrapi.httpx.get", return_value=_response(status, {"error": "nope"})):
        result = _tools().get_user("agno_agi")
    assert result.startswith(f"Error: TwittrAPI returned {status}")
    assert hint in result
    assert result.endswith(": nope")


def test_network_errors_are_returned_to_the_agent():
    with patch("agno.tools.twittrapi.httpx.get", side_effect=httpx.ConnectError("down")):
        assert _tools().get_trends() == "Error: TwittrAPI request failed: down"


def test_output_is_truncated_but_cursor_survives():
    big = {"data": [{"id": str(i), "text": "x" * 200} for i in range(20)], "nextCursor": "C2", "hasNextPage": True}
    with patch("agno.tools.twittrapi.httpx.get", return_value=_response(body=big)):
        result = _tools(max_chars=500).search_tweets("agno")
    assert result.startswith("next_cursor: C2")
    assert "[TRUNCATED: showing the first 500" in result


def test_custom_base_url():
    with patch("agno.tools.twittrapi.httpx.get", return_value=_response(body=USER)) as mock_get:
        _tools(base_url="https://example.test/").get_user("agno_agi")
    assert mock_get.call_args.args[0] == "https://example.test/user/by/username/agno_agi"


@pytest.mark.asyncio
async def test_async_search():
    client = MagicMock()
    client.get = AsyncMock(return_value=_response())
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    with patch("agno.tools.twittrapi.httpx.AsyncClient", return_value=client):
        result = await _tools().asearch_tweets("agno", category="Top")
    assert client.get.call_args.kwargs["params"]["category"] == "Top"
    assert result.startswith("next_cursor: CURSOR")


@pytest.mark.asyncio
async def test_async_error():
    client = MagicMock()
    client.get = AsyncMock(return_value=_response(402, {"error": "no balance"}))
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    with patch("agno.tools.twittrapi.httpx.AsyncClient", return_value=client):
        result = await _tools().aget_user_tweets(username="agno_agi")
    assert result.startswith("Error: TwittrAPI returned 402")
