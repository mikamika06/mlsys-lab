Our infrastructure engineering team is facing an escalating cloud expenditure for our large language model inference services, prompting leadership to demand a rigorous cost-benefit evaluation between continuing with our commercial per-token public API provider or migrating to a self-hosted vLLM architecture on dedicated GPU instances. 

To make a well-informed infrastructure decision, our finance partners require a robust quantitative model that addresses three critical cost dimensions. First, we need to precisely compute the break-even request volume where the fixed monthly leasing cost of self-hosting combined with lower variable compute costs undercuts the straight per-token public API pricing model, handling edge cases where self-hosting is never economically viable. 

Second, to optimize our self-hosted deployment, our systems team proposes utilizing cloud spot instances which can offer significant cost reductions of up to seventy percent, albeit with the constant risk of sudden preemption. We need a reliable mathematical formulation that calculates the expected cost of spot execution by accounting for hourly rates, preemption probabilities, restart overhead delays, and the precise economic loss of uncompleted work when an instance is abruptly terminated mid-task. 

Third, our workload extensively reuses shared context prompts, system instructions, and standard document headers across thousands of concurrent queries. We must accurately quantify the dollar savings achieved by enabling vLLM automatic prefix caching given a known cache hit rate, input token mix, and discounted token pricing structure. 

Your task is to implement the complete Python cost-modeling library that satisfies all these requirements with high precision and clean numerical stability.
