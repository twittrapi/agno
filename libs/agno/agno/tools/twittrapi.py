import json
from os import getenv
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple
from urllib.parse import quote

import httpx

from agno.tools import Toolkit
from agno.utils.log import log_debug, log_error

DEFAULT_BASE_URL = "https://api.twittrapi.com"

SearchCategory = Literal["Top", "Latest", "People", "Photos", "Videos"]

# What the API's status codes mean for the agent (https://www.twittrapi.com/docs/helpful-infos/error-handling).
STATUS_HINTS = {
    401: "the TwittrAPI key is missing or invalid",
    402: "the TwittrAPI balance is empty; top up at https://www.twittrapi.com",
    404: "not found",
    422: "the request was rejected; check the IDs, handle or search operators",
    429: "rate limited; wait a moment and retry",
}


class TwittrAPITools(Toolkit):
    """Read public X (Twitter) data through TwittrAPI.

    TwittrAPI (https://www.twittrapi.com) is a pay-per-call REST API for public X data, so no X
    developer account is needed. The toolkit covers search with X search operators, profiles,
    a user's tweets, followers and following, a tweet with its replies, and trends.

    Results use the API's compact ``format=simple`` JSON (flat camelCase tweets and users).
    Paginated functions put ``next_cursor`` on the first line, outside the ``max_chars`` budget,
    so the agent can ask for the next page even when the JSON is cut.

    Requests go through ``httpx``, already an Agno dependency, so this toolkit needs no
    additional install. Every successful call is billed to the TwittrAPI balance.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        enable_search_tweets: bool = True,
        enable_get_user: bool = True,
        enable_get_user_tweets: bool = True,
        enable_get_followers: bool = True,
        enable_get_following: bool = True,
        enable_get_tweet: bool = True,
        enable_get_trends: bool = True,
        all: bool = False,
        max_chars: int = 20000,
        timeout: int = 60,
        **kwargs,
    ):
        """Initialize TwittrAPITools.

        Args:
            api_key: TwittrAPI key, sent as the X-API-KEY header. Falls back to the
                TWITTRAPI_API_KEY env var.
            base_url: API base URL. Defaults to https://api.twittrapi.com.
            enable_search_tweets: Register search_tweets. Defaults to True.
            enable_get_user: Register get_user. Defaults to True.
            enable_get_user_tweets: Register get_user_tweets. Defaults to True.
            enable_get_followers: Register get_followers. Defaults to True.
            enable_get_following: Register get_following. Defaults to True.
            enable_get_tweet: Register get_tweet. Defaults to True.
            enable_get_trends: Register get_trends. Defaults to True.
            all: Register all functions. Defaults to False.
            max_chars: Most characters of JSON returned to the agent; longer output is cut with
                a marker. Defaults to 20000.
            timeout: Request timeout in seconds. Defaults to 60.
            **kwargs: Additional arguments passed to Toolkit.
        """
        self.api_key: Optional[str] = api_key or getenv("TWITTRAPI_API_KEY")
        if not self.api_key:
            log_error("TWITTRAPI_API_KEY not set. Please set the TWITTRAPI_API_KEY environment variable.")
        self.base_url: str = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.max_chars: int = max_chars
        self.timeout: int = timeout

        candidates: List[Tuple[bool, Callable[..., Any], Callable[..., Any]]] = [
            (enable_search_tweets, self.search_tweets, self.asearch_tweets),
            (enable_get_user, self.get_user, self.aget_user),
            (enable_get_user_tweets, self.get_user_tweets, self.aget_user_tweets),
            (enable_get_followers, self.get_followers, self.aget_followers),
            (enable_get_following, self.get_following, self.aget_following),
            (enable_get_tweet, self.get_tweet, self.aget_tweet),
            (enable_get_trends, self.get_trends, self.aget_trends),
        ]
        tools: List[Any] = []
        # Async variants are registered under the same names, for agent.arun() and agent.aprint_response().
        async_tools: List[Tuple[Callable[..., Any], str]] = []
        for enabled, sync_fn, async_fn in candidates:
            if enabled or all:
                tools.append(sync_fn)
                async_tools.append((async_fn, sync_fn.__name__))

        super().__init__(name="twittrapi_tools", tools=tools, async_tools=async_tools, **kwargs)

    # ------------------------------------------------------------------
    # Request building
    # ------------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        return {"X-API-KEY": self.api_key or "", "Accept": "application/json"}

    @staticmethod
    def _clean(params: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = {k: v for k, v in params.items() if v is not None and v != ""}
        cleaned["format"] = "simple"
        return cleaned

    @staticmethod
    def _handle(username: str) -> str:
        return username.strip().lstrip("@").strip()

    def _user_timeline(
        self, suffix: str, username: Optional[str], user_id: Optional[str], count: Optional[int], cursor: Optional[str]
    ) -> Tuple[str, Dict[str, Any]]:
        """Timelines take a user ID in the path; with only a handle, the API resolves
        ``/user/-1/...?username=<handle>`` itself."""
        if user_id:
            return f"/user/{quote(str(user_id), safe='')}/{suffix}", {"count": count, "cursor": cursor}
        if username and self._handle(username):
            return f"/user/-1/{suffix}", {"username": self._handle(username), "count": count, "cursor": cursor}
        raise ValueError("Provide either username or user_id.")

    def _search_request(
        self, query: str, category: Optional[str], count: Optional[int], cursor: Optional[str]
    ) -> Tuple[str, Dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("query must not be empty.")
        return "/search", {"searchItem": query, "category": category or "Latest", "count": count, "cursor": cursor}

    def _user_request(self, username: str) -> Tuple[str, Dict[str, Any]]:
        handle = self._handle(username or "")
        if not handle:
            raise ValueError("username must not be empty.")
        return f"/user/by/username/{quote(handle, safe='')}", {}

    def _tweet_request(self, tweet_id: str, cursor: Optional[str]) -> Tuple[str, Dict[str, Any]]:
        if not str(tweet_id or "").strip().isdigit():
            raise ValueError("tweet_id must be a numeric tweet ID.")
        return f"/tweet/{str(tweet_id).strip()}", {"cursor": cursor}

    def _trends_request(self, location_id: int) -> Tuple[str, Dict[str, Any]]:
        return f"/trends/{int(location_id)}", {}

    # ------------------------------------------------------------------
    # Transport and output
    # ------------------------------------------------------------------

    def _missing_key(self) -> Optional[str]:
        if not self.api_key:
            return "Error: TWITTRAPI_API_KEY is not set. Create a key at https://www.twittrapi.com."
        return None

    def _get(self, path: str, params: Dict[str, Any], paginated: bool) -> str:
        missing = self._missing_key()
        if missing:
            return missing
        try:
            log_debug(f"TwittrAPI GET {path}")
            response = httpx.get(
                f"{self.base_url}{path}", params=self._clean(params), headers=self._headers(), timeout=self.timeout
            )
        except httpx.HTTPError as e:
            log_error(f"TwittrAPI request failed: {e}")
            return f"Error: TwittrAPI request failed: {e}"
        return self._format(response, paginated)

    async def _aget(self, path: str, params: Dict[str, Any], paginated: bool) -> str:
        missing = self._missing_key()
        if missing:
            return missing
        try:
            log_debug(f"TwittrAPI GET {path}")
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_url}{path}", params=self._clean(params), headers=self._headers()
                )
        except httpx.HTTPError as e:
            log_error(f"TwittrAPI request failed: {e}")
            return f"Error: TwittrAPI request failed: {e}"
        return self._format(response, paginated)

    def _format(self, response: httpx.Response, paginated: bool) -> str:
        if response.status_code >= 400:
            message = ""
            try:
                body = response.json()
                if isinstance(body, dict):
                    message = str(body.get("error") or "")
            except ValueError:
                message = response.text[:200]
            hint = STATUS_HINTS.get(response.status_code, "request failed")
            log_error(f"TwittrAPI error {response.status_code}: {message or hint}")
            return f"Error: TwittrAPI returned {response.status_code} ({hint}){': ' + message if message else ''}"

        try:
            data = response.json()
        except ValueError:
            return "Error: TwittrAPI returned a response that is not JSON."

        if isinstance(data, dict) and isinstance(data.get("data"), list):
            # Tweets X no longer returns (deleted, withheld) come back as {"id": ..., "unavailable": true}.
            data["data"] = [row for row in data["data"] if not (isinstance(row, dict) and row.get("unavailable"))]

        text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        if len(text) > self.max_chars:
            text = (
                text[: self.max_chars]
                + f"\n\n[TRUNCATED: showing the first {self.max_chars} of {len(text)} characters. Request a smaller "
                "count or use next_cursor to page.]"
            )
        if not paginated or not isinstance(data, dict):
            return text
        cursor = data.get("nextCursor") if data.get("hasNextPage") else None
        if not cursor:
            return f"next_cursor: none (last page)\n\n{text}"
        return f"next_cursor: {cursor}\nPass next_cursor as cursor to get the next page.\n\n{text}"

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def search_tweets(
        self,
        query: str,
        category: Optional[SearchCategory] = None,
        count: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> str:
        """Use this function to search X (Twitter) like the X search box.

        The query supports X search operators, e.g. from:user, to:user, "exact phrase", lang:en,
        since:2026-01-01, until:2026-02-01, min_faves:100, filter:media, -filter:replies.

        Args:
            query (str): Search query in X search syntax.
            category (Optional[str]): "Latest" (default, newest first), "Top", "People" (accounts
                instead of tweets), "Photos" or "Videos".
            count (Optional[int]): Results per page, 1 to 100 (default 20).
            cursor (Optional[str]): next_cursor from a previous result, to get the next page.

        Returns:
            str: JSON with the tweets (or accounts), with next_cursor on the first line.
        """
        try:
            path, params = self._search_request(query, category, count, cursor)
        except ValueError as e:
            return f"Error: {e}"
        return self._get(path, params, paginated=True)

    async def asearch_tweets(
        self,
        query: str,
        category: Optional[SearchCategory] = None,
        count: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> str:
        """Use this function to search X (Twitter) like the X search box.

        The query supports X search operators, e.g. from:user, to:user, "exact phrase", lang:en,
        since:2026-01-01, until:2026-02-01, min_faves:100, filter:media, -filter:replies.

        Args:
            query (str): Search query in X search syntax.
            category (Optional[str]): "Latest" (default, newest first), "Top", "People" (accounts
                instead of tweets), "Photos" or "Videos".
            count (Optional[int]): Results per page, 1 to 100 (default 20).
            cursor (Optional[str]): next_cursor from a previous result, to get the next page.

        Returns:
            str: JSON with the tweets (or accounts), with next_cursor on the first line.
        """
        try:
            path, params = self._search_request(query, category, count, cursor)
        except ValueError as e:
            return f"Error: {e}"
        return await self._aget(path, params, paginated=True)

    def get_user(self, username: str) -> str:
        """Use this function to get an X (Twitter) profile by handle.

        Returns name, bio, location, follower/following/tweet counts, verification, created date
        and the numeric user ID.

        Args:
            username (str): X handle, e.g. "XDevelopers" (a leading @ is fine).

        Returns:
            str: JSON with the profile, or data null when the handle does not exist.
        """
        try:
            path, params = self._user_request(username)
        except ValueError as e:
            return f"Error: {e}"
        return self._get(path, params, paginated=False)

    async def aget_user(self, username: str) -> str:
        """Use this function to get an X (Twitter) profile by handle.

        Returns name, bio, location, follower/following/tweet counts, verification, created date
        and the numeric user ID.

        Args:
            username (str): X handle, e.g. "XDevelopers" (a leading @ is fine).

        Returns:
            str: JSON with the profile, or data null when the handle does not exist.
        """
        try:
            path, params = self._user_request(username)
        except ValueError as e:
            return f"Error: {e}"
        return await self._aget(path, params, paginated=False)

    def get_user_tweets(
        self,
        username: Optional[str] = None,
        user_id: Optional[str] = None,
        count: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> str:
        """Use this function to get a user's recent tweets, newest first, without replies.

        Args:
            username (Optional[str]): X handle. Give this or user_id.
            user_id (Optional[str]): Numeric X user ID (a little faster than username).
            count (Optional[int]): Tweets per page, 1 to 100 (default 20).
            cursor (Optional[str]): next_cursor from a previous result, to get the next page.

        Returns:
            str: JSON with the tweets, with next_cursor on the first line.
        """
        try:
            path, params = self._user_timeline("tweets", username, user_id, count, cursor)
        except ValueError as e:
            return f"Error: {e}"
        return self._get(path, params, paginated=True)

    async def aget_user_tweets(
        self,
        username: Optional[str] = None,
        user_id: Optional[str] = None,
        count: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> str:
        """Use this function to get a user's recent tweets, newest first, without replies.

        Args:
            username (Optional[str]): X handle. Give this or user_id.
            user_id (Optional[str]): Numeric X user ID (a little faster than username).
            count (Optional[int]): Tweets per page, 1 to 100 (default 20).
            cursor (Optional[str]): next_cursor from a previous result, to get the next page.

        Returns:
            str: JSON with the tweets, with next_cursor on the first line.
        """
        try:
            path, params = self._user_timeline("tweets", username, user_id, count, cursor)
        except ValueError as e:
            return f"Error: {e}"
        return await self._aget(path, params, paginated=True)

    def get_followers(
        self,
        username: Optional[str] = None,
        user_id: Optional[str] = None,
        count: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> str:
        """Use this function to get the accounts that follow a user.

        Args:
            username (Optional[str]): X handle. Give this or user_id.
            user_id (Optional[str]): Numeric X user ID.
            count (Optional[int]): Accounts per page, 1 to 100 (default 20).
            cursor (Optional[str]): next_cursor from a previous result, to get the next page.

        Returns:
            str: JSON with the follower profiles, with next_cursor on the first line.
        """
        try:
            path, params = self._user_timeline("followers", username, user_id, count, cursor)
        except ValueError as e:
            return f"Error: {e}"
        return self._get(path, params, paginated=True)

    async def aget_followers(
        self,
        username: Optional[str] = None,
        user_id: Optional[str] = None,
        count: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> str:
        """Use this function to get the accounts that follow a user.

        Args:
            username (Optional[str]): X handle. Give this or user_id.
            user_id (Optional[str]): Numeric X user ID.
            count (Optional[int]): Accounts per page, 1 to 100 (default 20).
            cursor (Optional[str]): next_cursor from a previous result, to get the next page.

        Returns:
            str: JSON with the follower profiles, with next_cursor on the first line.
        """
        try:
            path, params = self._user_timeline("followers", username, user_id, count, cursor)
        except ValueError as e:
            return f"Error: {e}"
        return await self._aget(path, params, paginated=True)

    def get_following(
        self,
        username: Optional[str] = None,
        user_id: Optional[str] = None,
        count: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> str:
        """Use this function to get the accounts a user follows.

        Args:
            username (Optional[str]): X handle. Give this or user_id.
            user_id (Optional[str]): Numeric X user ID.
            count (Optional[int]): Accounts per page, 1 to 100 (default 20).
            cursor (Optional[str]): next_cursor from a previous result, to get the next page.

        Returns:
            str: JSON with the followed profiles, with next_cursor on the first line.
        """
        try:
            path, params = self._user_timeline("followings", username, user_id, count, cursor)
        except ValueError as e:
            return f"Error: {e}"
        return self._get(path, params, paginated=True)

    async def aget_following(
        self,
        username: Optional[str] = None,
        user_id: Optional[str] = None,
        count: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> str:
        """Use this function to get the accounts a user follows.

        Args:
            username (Optional[str]): X handle. Give this or user_id.
            user_id (Optional[str]): Numeric X user ID.
            count (Optional[int]): Accounts per page, 1 to 100 (default 20).
            cursor (Optional[str]): next_cursor from a previous result, to get the next page.

        Returns:
            str: JSON with the followed profiles, with next_cursor on the first line.
        """
        try:
            path, params = self._user_timeline("followings", username, user_id, count, cursor)
        except ValueError as e:
            return f"Error: {e}"
        return await self._aget(path, params, paginated=True)

    def get_tweet(self, tweet_id: str, cursor: Optional[str] = None) -> str:
        """Use this function to get a tweet by ID with its reply thread.

        Returns the full text, author, media, engagement counts and the replies in display order.

        Args:
            tweet_id (str): Numeric tweet ID (the number at the end of an x.com/.../status/ URL).
            cursor (Optional[str]): next_cursor from a previous result, to load more replies.

        Returns:
            str: JSON with the tweet and its replies, with next_cursor on the first line.
        """
        try:
            path, params = self._tweet_request(tweet_id, cursor)
        except ValueError as e:
            return f"Error: {e}"
        return self._get(path, params, paginated=True)

    async def aget_tweet(self, tweet_id: str, cursor: Optional[str] = None) -> str:
        """Use this function to get a tweet by ID with its reply thread.

        Returns the full text, author, media, engagement counts and the replies in display order.

        Args:
            tweet_id (str): Numeric tweet ID (the number at the end of an x.com/.../status/ URL).
            cursor (Optional[str]): next_cursor from a previous result, to load more replies.

        Returns:
            str: JSON with the tweet and its replies, with next_cursor on the first line.
        """
        try:
            path, params = self._tweet_request(tweet_id, cursor)
        except ValueError as e:
            return f"Error: {e}"
        return await self._aget(path, params, paginated=True)

    def get_trends(self, location_id: int = 1) -> str:
        """Use this function to get the trending topics on X for a location.

        Args:
            location_id (int): WOEID of the location: 1 is worldwide (default), 23424977 the
                United States, 23424975 the United Kingdom, 23424829 Germany.

        Returns:
            str: JSON with the trends.
        """
        return self._get(*self._trends_request(location_id), paginated=False)

    async def aget_trends(self, location_id: int = 1) -> str:
        """Use this function to get the trending topics on X for a location.

        Args:
            location_id (int): WOEID of the location: 1 is worldwide (default), 23424977 the
                United States, 23424975 the United Kingdom, 23424829 Germany.

        Returns:
            str: JSON with the trends.
        """
        return await self._aget(*self._trends_request(location_id), paginated=False)
