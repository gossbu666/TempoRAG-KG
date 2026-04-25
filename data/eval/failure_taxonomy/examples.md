# Failure Examples

One representative prediction per category (closest to the median F1 within that bucket).

## A1

- **Question:** What was the percentage of META's revenue allocated to general and administrative expenses in 2023, as disclosed in the 10-K filing signed by Mark Zuckerberg, and how did this compare to the prior year's percentage?
- **Gold:** In 2023, META's general and administrative expenses accounted for 8% of revenue, as disclosed in the 10-K filing signed by CEO Mark Zuckerberg (Chunk 2). This represented a 2 percentage-point decrease from 2022, where these expenses were 10% of revenue. The reduction reflects improved cost efficiency under the leadership structure overseen by the executive team (Chunk 1 → Chunk 2).
- **Prediction:** `In 2023, META's general and administrative expenses were $11,408 million out of $134,902 million in revenue, which is approximately 8.46%. In 2022, these expenses were $11,816 million out of $116,609 million in revenue, approximately 10.13%.`
- **F1:** 0.35555555555555557
- **Scope / hop:** intra / 2
- **Condition × model:** L1 / gpt-4o
- **Classifier rationale:** The gold answer's specific percentage allocation was not found in any of the retrieved chunks.

## A2

- **Question:** How many full-time and part-time employees did Amazon have as of the end of fiscal year 2023?
- **Gold:** Amazon reported approximately 1,525,000 full-time and part-time employees as of December 31, 2023 (excluding contractors and temporary workers).
- **Prediction:** `Approximately 1,525,000 employees`
- **F1:** 0.2857142857142857
- **Scope / hop:** intra / 1
- **Condition × model:** L1 / gpt-4.1-nano
- **Classifier rationale:** The model's prediction is incomplete; it lacks the distinction between full-time and part-time employees.

## A4

- **Question:** What was the dollar increase in Cisco's APJC revenue from 2023 to 2024, and how does this growth relate to Charles H. Robbins' executive responsibilities as disclosed in Cisco's filings?
- **Gold:** Cisco's APJC revenue increased from $5,382 million in 2023 to $8,208 million in 2024, representing a $2,826 million growth (Chunk 2). Charles H. Robbins, as Cisco's Chair and CEO, held full authority over corporate strategy and financial decision-making (Chunk 1). This revenue surge directly reflects the strategic leadership and operational oversight responsibilities Robbins maintained as chief executive, demonstrating how executive stewardship translates to regional financial performance.
- **Prediction:** `I don't know`
- **F1:** 0.0
- **Scope / hop:** inter_year / 2
- **Condition × model:** L2 / gpt-4.1-nano

## A5

- **Question:** What was the dollar decrease in Cisco's total long-lived assets from 2021 to 2022, and how does this relate to R. Scott Herren's role as CFO in certifying financial disclosures?
- **Gold:** Cisco's total long-lived assets decreased by $432 million from $3,433 million in 2021 to $3,001 million in 2022 (Chunk 2). As CFO, R. Scott Herren is responsible for certifying the accuracy of these financial disclosures in the 10-K filing (Chunk 1), ensuring stakeholders that the reported asset values comply with regulatory standards and reflect the company's financial position.
- **Prediction:** `I don't know`
- **F1:** 0.0
- **Scope / hop:** inter_year / 2
- **Condition × model:** L2 / gpt-oss-120b

## B2

- **Question:** What partnership did Microsoft describe with OpenAI in its fiscal 2023 10-K filing, and what is the forward-looking implication?
- **Gold:** Microsoft disclosed a multi-year, multi-billion dollar investment partnership with OpenAI that includes deep technology integration (Azure as the exclusive cloud provider for OpenAI) and Microsoft's AI offerings being powered by OpenAI technology. The forward-looking implication is that Microsoft expects AI capabilities — including Copilot across its product suite — to become material to future revenue; specific future revenue contribution is not quantified in the filing.
- **Prediction:** `Microsoft has a long-term partnership with OpenAI, a leading AI research and deployment company, where Azure powers all of OpenAI's workloads, and the forward-looking implication is that this partnership will help Microsoft accelerate its development and deployment of AI technologies.`
- **F1:** 0.28571428571428575
- **Scope / hop:** forward_looking / 2
- **Condition × model:** L3 / llama-70b

## B3

- **Question:** What was Oracle's total revenue for the fiscal year ending May 31, 2024? Note that Oracle's fiscal year does not align with the calendar year.
- **Gold:** Oracle reported total revenues of $52,961 million for fiscal year 2024, which ended May 31, 2024. Oracle's fiscal 2024 spans June 1, 2023 to May 31, 2024.
- **Prediction:** `$52,961 million`
- **F1:** 0.13793103448275862
- **Scope / hop:** fiscal_vs_calendar / 1
- **Condition × model:** L2 / gpt-4.1-nano

## B5

- **Question:** How does the audit opinion from Ernst & Young LLP on Intel's internal controls influence the transparency of corporate infrastructure cost disclosures, and what does this reveal about similar segment reporting practices in NVIDIA's financial statements?
- **Gold:** Ernst & Young LLP's unqualified audit opinion (Chunk 1) confirms Intel maintains effective internal controls over financial reporting, ensuring reliability in segment disclosures. This validation allows Intel's 'All Other' segment (Chunk 2) to transparently include unallocated corporate infrastructure costs as part of its financial reporting framework. Similarly, NVIDIA's 'All Other' category (Chunk 3) explicitly discloses corporate infrastructure and support costs, demonstrating that rigorous audit processes across both companies enable standardized reporting of enterprise-level expenses within comparable segments. This chain shows how audit rigor (Chunk 1) directly supports granular segment transparency (Chunk 2), which then aligns with industry-wide disclosure practices (Chunk 3).
- **Prediction:** `The audit opinion states that Intel maintained effective internal control over financial reporting as of December 28, 2024, which supports transparency in disclosures, but it does not provide specific details about corporate infrastructure costs or segment reporting practices in NVIDIA's financial statements.`
- **F1:** 0.3404255319148936
- **Scope / hop:** cross_company / 3
- **Condition × model:** L3 / gpt-4.1-nano

## NF

- **Question:** How many full-time and part-time employees did Amazon have as of the end of fiscal year 2023?
- **Gold:** Amazon reported approximately 1,525,000 full-time and part-time employees as of December 31, 2023 (excluding contractors and temporary workers).
- **Prediction:** `Approximately 1,525,000 full-time and part-time employees.`
- **F1:** 0.5
- **Scope / hop:** intra / 1
- **Condition × model:** L2 / gpt-4.1-nano

