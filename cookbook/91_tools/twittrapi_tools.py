"""
TwittrAPI Tools
=============================

Demonstrates reading public X (Twitter) data with TwittrAPI: search with X search
operators, profiles, a user's tweets, followers, a tweet with its replies, and trends.

TwittrAPI is a pay-per-call REST API, so no X developer account is needed.
Set TWITTRAPI_API_KEY (create a key at https://www.twittrapi.com) before running.
"""

import asyncio

from agno.agent import Agent
from agno.tools.twittrapi import TwittrAPITools

# ---------------------------------------------------------------------------
# Create Agent
# ---------------------------------------------------------------------------

# Example 1: all read tools
agent = Agent(
    tools=[TwittrAPITools()],
    instructions=[
        "Use search_tweets with X search operators (from:, lang:, min_faves:, -filter:replies) to find tweets.",
        "Quote the tweet URL for every tweet you cite.",
    ],
)

# Example 2: search only, smaller outputs
search_agent = Agent(
    tools=[
        TwittrAPITools(
            enable_get_user=False,
            enable_get_user_tweets=False,
            enable_get_followers=False,
            enable_get_following=False,
            enable_get_tweet=False,
            enable_get_trends=False,
            max_chars=8000,
        )
    ]
)

# ---------------------------------------------------------------------------
# Run Agent
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    agent.print_response(
        "What has @XDevelopers posted this week? Summarize the main topics.",
        markdown=True,
    )

    search_agent.print_response(
        "Find popular English tweets about AI agents from the last 3 days (min_faves:50) and group them by theme.",
        markdown=True,
    )

    agent.print_response("What is trending worldwide on X right now?", markdown=True)

    # agent.aprint_response() runs the toolkit's async variants
    asyncio.run(
        agent.aprint_response(
            "Look up @XDevelopers and describe a few of their recent followers.",
            markdown=True,
        )
    )
