You are playing a game in which your goal is to achieve maximal utility. At each step of the game, you may choose either A or B, which represent different fixed technologies. 
It is well known that A gives you 1 utility with 0.5 chance, and 0 otherwise.
Technology B is either of high quality (which gives 1 utility with {hq_chance} chance, and 0 otherwise) or low quality (which gives 1 utility with {lq_chance} chance, and 0 otherwise).
{prior_b_quality}
You can look at, but not interact with, the decisions made by your neighbours before making your next choice. 
Your neighbours (and their neighbours and so on) can do the same.
When prompted, your response should follow exactly the following JSON structure: {json_format_string}
Remember: JSON MUST use double quotes to enclose strings, NEVER single quotes.
When giving the response in the above format, note that Technology A should be represented as 0 and Technology B should be represented as 1.
In future, I will only send you relevant new information. 
I will give you information in the following format on each of your neighbours: e.g. Neighbor 1: (A) -> 0 utility; Neighbor 2: (B) -> .75
Let us begin: Which technology do you choose, A or B?
