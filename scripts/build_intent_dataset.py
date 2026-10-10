#!/usr/bin/env python3
"""
scripts/build_intent_dataset.py

Generates and partitions the gold-standard Intent Classification Dataset of 400
annotated student utterances across 6 system routes, strictly adhering to spec.md Section 2.3.

Target Intents:
1. check_calendar_date   (70 queries) -> Deterministic SQLite
2. check_module_prereq   (75 queries) -> Deterministic SQLite
3. query_general_policy  (90 queries) -> Unstructured Hybrid RAG
4. escalate_human_contact(55 queries) -> Deterministic SQLite / Handoff
5. clarification_response(50 queries) -> Multi-turn Slot Payload
6. out_of_scope          (60 queries) -> OOD Guardrail / Rejection

Splits (Stratified, seed=42):
- 70% Train (280)
- 15% Validation (60)
- 15% Test (60)

Outputs:
- data/intents/intent_dataset.json (Unified dataset with split markers)
- data/intents/train.json
- data/intents/val.json
- data/intents/test.json
"""

import json
import os
import random
from typing import Dict, List

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "intents")

# 1. CHECK_CALENDAR_DATE (70 queries)
CALENDAR_QUERIES = [
    "When do lectures commence for Semester 1 2026?",
    "What is the start date of Semester 2 classes?",
    "When does the midterm break start in semester 1?",
    "Dates for the April student recess 2026",
    "When do first semester lectures end?",
    "What is the date for the mid-semester test week?",
    "When is the June 2026 examination period?",
    "Start date of first opportunity exams in semester 1",
    "When do second opportunity examinations take place?",
    "What is the registration deadline for new students?",
    "When is the last day to add or drop courses without penalty in semester 1?",
    "When does late registration end for semester 2?",
    "Graduation ceremony dates for April 2026",
    "When is the October graduation congregation?",
    "What day does classes resume after the mid-term recess?",
    "When are the institutional holidays in 2026?",
    "Is the university open on Cassinga Day?",
    "When is the deadline for thesis submission for postgrads in 2026?",
    "When does hostel move-in begin for first years?",
    "What is the date for student orientation week?",
    "When do first semester exam results come out?",
    "Release date for June 2026 examination marks",
    "What date is the commencement of second semester lectures?",
    "When does the September spring recess start?",
    "Last day of classes for semester 2",
    "When do end-of-year examinations start?",
    "Dates for November 2026 final exams",
    "When is the deadline to change study mode from full-time to distance?",
    "What is the deadline date for semester 1 course cancellation with 100% credit?",
    "When is the 50% tuition fee credit deadline for semester 2?",
    "Date for commencement of supplementary exams in July",
    "When does the university close for December holidays?",
    "What is the calendar date for the NUST Cultural Festival?",
    "When are the SRC election dates held?",
    "When does registration for 2027 prospective students open?",
    "Date for late registration penalty enforcement",
    "When do part-time evening lectures start?",
    "Is there class during the test week?",
    "When is the first opportunity exam timetable released?",
    "Check date of first semester test week",
    "When is the deadline to pay the second fee installment?",
    "What day does the semester 2 recess end?",
    "When is the final date for submission of continuous assessment marks?",
    "Calendar date for commencement of lectures",
    "When is the last day for international student registration?",
    "When do re-examinations take place?",
    "Check date for April graduation session",
    "When do winter classes start?",
    "What is the date for Africa Day holiday observance?",
    "When does term 1 end?",
    "Tell me the exact dates for semester 1 recess",
    "When is the deadline to appeal continuous assessment marks?",
    "What date does the academic year 2026 officially start?",
    "When is the deadline to register research topics for master students?",
    "Check calendar for exam opportunity dates",
    "When do classes finish before study week?",
    "What day is study break before June exams?",
    "When is the deadline to submit medical certificates for missed tests?",
    "Date of semester 2 test week 2026",
    "When are special examinations scheduled in the calendar?",
    "What is the date for Ascension Day holiday?",
    "When do hostel residences close for the mid-year break?",
    "What day do hostels re-open for semester 2?",
    "When is the final date for remark application submissions?",
    "Check dates for November second opportunity exams",
    "When does the first term lecture period officially conclude?",
    "What date is the second installment of tuition fees due?",
    "When do postgraduate coursework classes start?",
    "When is the deadline for exemption application submissions in semester 1?",
    "Check academic calendar for term start and end dates"
]

# 2. CHECK_MODULE_PREREQ (75 queries)
PREREQ_QUERIES = [
    "What are the prerequisites for Data Structures and Algorithms DSA611S?",
    "Can I register for DSA611S if I failed Programming 1 PRG510S?",
    "What do I need to take Object Oriented Programming OOP611S?",
    "Prerequisites for Software Engineering SWE712S",
    "What are the corequisites for Web Development Fundamentals WDF521S?",
    "Can I enroll in Advanced Database Systems ADB711S without Database Systems DBS611S?",
    "What are the prerequisites for Artificial Intelligence ART811S?",
    "Tell me the prerequisite chain for Distributed Systems DST711S",
    "What modules do I need before taking Computer Networks 1 CMP611S?",
    "Is Introduction to Business Management a prerequisite for Principles of Marketing?",
    "What are the prerequisites for Financial Accounting 1A FAC511S?",
    "Can I register for Financial Accounting 1B without passing 1A?",
    "What do I need to take Structural Analysis 1 STA611S?",
    "Prerequisites for Fluid Mechanics FLM621S in engineering",
    "What are the prerequisites for Engineering Mathematics 2?",
    "Can I do Engineering Mathematics 3 if I only got 48 in Math 2?",
    "What modules are required before doing Work-Integrated Learning WIL711S?",
    "Prerequisites for Bachelor of Computer Science capstone project",
    "What is required to register for Honours Research Project in FCI?",
    "Tell me the corequisites for Digital Logic Design DLD511S",
    "What are the prerequisites for Operating Systems OPS612S?",
    "Can I take PRG621S without completing PRG510S?",
    "What modules lead into Machine Learning MLG812S?",
    "Prerequisites for Mobile Application Development MAD612S",
    "What do I need before taking Information Security IFS711S?",
    "What are the prerequisites for Thermodynamics in FEBE?",
    "Requirements to register for Concrete Technology in civil engineering",
    "What are the prerequisites for Microeconomics 1?",
    "Can I take Macroeconomics 2 if I failed Microeconomics 1?",
    "Prerequisites for Company Law in the Commerce faculty",
    "What modules do I need to pass before registering for Auditing 2?",
    "Can I register for Commercial Law 1A and 1B at the same time?",
    "Prerequisites for Emergency Care Clinical Practice 1 in health sciences",
    "What do I need to register for Human Anatomy and Physiology 2?",
    "Prerequisites for Clinical Pharmacology 1",
    "Can I register for Medical Biochemistry without General Chemistry?",
    "What are the prerequisites for Environmental Law in natural resources?",
    "What modules do I need to complete before Land Law 1?",
    "Prerequisites for Urban Planning Theory in FEBE",
    "Can I take Highway Engineering without Surveying 1?",
    "What do I need to register for Electrical Machines 1?",
    "Prerequisites for Power Systems Analysis in electrical engineering",
    "Can I take Network Security without Computer Networks 2?",
    "What are the prerequisite courses for Business Statistics?",
    "Prerequisites for Quantitative Methods in GSB",
    "What modules do I need to pass before doing the MBA dissertation?",
    "Is Introduction to Psychology a prerequisite for Organizational Behaviour?",
    "Prerequisites for Journalism Practice 2 in communication",
    "What do I need before taking Broadcast Journalism?",
    "Prerequisites for Translation Theory and Practice",
    "Can I take German 2 without German 1A?",
    "Prerequisites for Logistics Management 2",
    "What modules do I need before taking Supply Chain Management?",
    "Can I register for Purchasing Management without Principles of Management?",
    "Prerequisites for Corporate Finance 1 in business school",
    "What do I need to take Investment Analysis?",
    "Prerequisites for Taxation 1 in accounting",
    "Can I take Cost and Management Accounting 2 without CMA 1?",
    "What are the prerequisite rules for Systems Analysis and Design?",
    "Prerequisites for Cloud Computing Architecture",
    "What do I need before taking Cyber Security Governance?",
    "Prerequisites for Highway Geometric Design",
    "Can I register for Soil Mechanics 2 without Geology?",
    "What are the prerequisites for Geotechnical Engineering?",
    "Requirements to take Renewable Energy Systems in mechanical engineering",
    "Prerequisites for Control Systems 1 in electrical engineering",
    "What do I need to take Cryptography and Network Security?",
    "Prerequisites for Big Data Analytics in data science",
    "Can I enroll in Mining Geology without Physical Geology?",
    "What are the prerequisites for Mine Ventilation in mining engineering?",
    "Prerequisites for Hydrology and Water Resources in civil engineering",
    "Can I take Structural Steel Design without Structural Analysis 2?",
    "What do I need to register for Embedded Systems 1?",
    "Prerequisites for Natural Language Processing in computing",
    "Check prerequisites for module SWE711S"
]

# 3. QUERY_GENERAL_POLICY (90 queries)
POLICY_QUERIES = [
    "What is the minimum attendance requirement to be allowed to sit for examinations?",
    "Can I write exams if my attendance is only 75%?",
    "What happens if I miss more than 20% of classes?",
    "What is the official policy on lecture attendance?",
    "What is the pass mark for a semester course at NUST?",
    "What continuous assessment mark do I need to qualify for exams?",
    "What is the examination subminimum requirement?",
    "Can I pass a course if my exam mark is below 40%?",
    "How does the supplementary examination qualification work?",
    "Who qualifies for a second opportunity or supplementary examination?",
    "What mark gets you a supplementary exam?",
    "What percentage qualifies for a pass with distinction?",
    "How do I qualify for an exit examination?",
    "Can I write an exit exam if I have two courses left to graduate?",
    "What are the rules and procedures for requesting an exam script remark?",
    "How much does it cost to have an exam paper remarked?",
    "What is the deadline to submit an application for exam remarking?",
    "Will I get my remarking fee refunded if my mark increases?",
    "What is the fee for a special examination due to medical reasons?",
    "How do I apply for a special exam if I was sick during the main exam?",
    "What is the academic exclusion policy at NUST?",
    "How many times can I repeat a failed module before being excluded?",
    "What is the N+2 rule for degree completion?",
    "How do I appeal against academic exclusion?",
    "What are the requirements for Mature Age Entry admission?",
    "Can I apply through mature age entry if I am 23 years old?",
    "What subjects are tested in the mature age entry examination?",
    "How do I apply for recognition of prior learning RPL?",
    "What is the application fee for recognition of prior learning?",
    "How do credit transfers and course exemptions work from other universities?",
    "Can I transfer my credits from UNAM to NUST?",
    "What is the maximum percentage of credits that can be exempted?",
    "How do I apply for resumption of studies after taking a gap semester?",
    "What are the hostel residence rules regarding visitors?",
    "What is the hostel curfew time for residential halls?",
    "How much is the breakage deposit for on-campus hostel accommodation?",
    "Can I cook inside my hostel bedroom?",
    "What meal plan is mandatory for hostel students?",
    "How much is the breakfast, lunch, and dinner voucher in the dining hall?",
    "What are the hostel rules for Shangri-La, Monresa, and Höpker residences?",
    "What is the registration deposit fee for undergraduate students?",
    "How much is the late registration penalty fee?",
    "What are the payment deadlines for semester tuition fees?",
    "What happens if I don't pay my student account by the due date?",
    "What is the refund policy if I cancel my registration?",
    "What percentage of fees is credited if I cancel courses in the second week?",
    "How do I set up a monthly debit order payment plan for my fees?",
    "Are international students required to pay an additional levy?",
    "What is the difference between SADC and non-SADC student fee levies?",
    "How does NSFAS funding work for paying tuition and accommodation at NUST?",
    "What are the official NUST bank details for paying student fees?",
    "Which bank account should I deposit my registration fee into?",
    "What reference must I use when making a bank transfer to NUST?",
    "What are the opening hours of the NUST main library?",
    "Is the library open on weekends and Sundays?",
    "What are the library operating hours during student vacations?",
    "How many books can an undergraduate student borrow from the library?",
    "What is the daily fine for an overdue library book?",
    "Where is the campus health clinic located?",
    "What time does the student clinic open on weekdays?",
    "How much is the consultation fee to visit the campus nurse?",
    "Does NUST provide free psychological counselling and mental health support?",
    "How do I make an appointment with the student counsellor?",
    "What are the powers and functions of the Students' Representative Council SRC?",
    "How do student societies register with the Department of Student Services?",
    "What are the gown hiring procedures for graduation ceremony?",
    "How do I get academic clearance for graduation?",
    "What is the policy on student academic misconduct and plagiarism?",
    "What disciplinary action is taken for cheating in a test or exam?",
    "Can a student be expelled for plagiarism in their thesis?",
    "What is the procedure to appeal a disciplinary committee decision?",
    "What are the regulations on student alcohol possession on campus premises?",
    "Are pets allowed in the student residences?",
    "What is the policy for dropping a course after the semester deadline?",
    "How do I request an official academic transcript or proof of registration?",
    "How much does a duplicate student card cost?",
    "What is the fee for a duplicate degree certificate?",
    "How do postgraduate students register their research thesis topics?",
    "What is the maximum allowed duration for completing a Master's degree?",
    "How many postgraduate students can a single supervisor take?",
    "What are the requirements for submitting a doctoral dissertation?",
    "What is the rule on changing programmes within the university?",
    "What is the minimum APS admission score for engineering programmes?",
    "How does the Cambridge A level grade convert to NUST admission points?",
    "Can I register for evening classes if I work full-time?",
    "What is the policy on deferring an examination to the next session?",
    "How do I apply for a refund of excess credit balance on my student account?",
    "What are the requirements for emergency medical care program selection?",
    "Explain the NUST code of conduct for student behavior",
    "What is the required mark to achieve a pass with distinction at NUST?"
]

# 4. ESCALATE_HUMAN_CONTACT (55 queries)
CONTACT_QUERIES = [
    "I need to speak to someone at the Katima Mulilo regional centre",
    "What is the telephone number for the Rundu campus office?",
    "How do I contact the Gobabis regional coordinator?",
    "Give me the email address for the Keetmanshoop centre",
    "Who is the coordinator for the Otjiwarongo regional office?",
    "Phone number for the Walvis Bay regional campus",
    "I need the contact details for Tsumeb regional centre",
    "How can I reach the Outapi resource centre coordinator?",
    "Email address for the Ongwediva resource centre",
    "Contact number for Eenhana satellite campus",
    "Who is the head of the Rietfontein satellite campus?",
    "Give me the phone number and email for the Lüderitz satellite campus",
    "I need the contact info for the Oranjemund resource centre",
    "Who is the faculty officer for Computing and Informatics?",
    "Give me the email address for the Faculty of Engineering FEBE office",
    "How do I contact the Faculty Officer for Commerce and Human Sciences?",
    "Contact details for the Faculty of Health and Natural Resources",
    "Email address for the Harold Pupkewitz Graduate School of Business office",
    "How do I reach the Examinations Department desk?",
    "What is the phone number for the Office of the Registrar?",
    "I need to talk to someone in Student Financial Services / Debtors",
    "Email address to submit proof of payment to student debtors",
    "Who is the Director of Student Services and what is his office number?",
    "How do I contact the Student Counsellor?",
    "What is the telephone number for the campus health clinic?",
    "Email address for the NUST Main Library reference desk",
    "How do I contact the Admissions Office directly?",
    "Give me the phone number for international student support",
    "Connect me with a human staff member please",
    "I want to speak with an advisor",
    "Can I talk to a real person from admissions?",
    "Who can I email about my registration hold?",
    "I need help from a human officer regarding my timetable clash",
    "Give me the contact person for hostel accommodation queries",
    "Who is the Superintendent for Shangri-La hostel?",
    "Phone number for campus security control room",
    "How do I reach the IT Helpdesk for portal login problems?",
    "Email address for student portal PIN reset support",
    "Who is the SRC President and how do I contact them?",
    "Contact details for the Land and Property Sciences student society",
    "Email for the NUST Accounting Society",
    "How do I contact the Computer Science student society?",
    "Who do I speak to regarding postgraduate scholarship applications?",
    "Faculty officer telephone extension for engineering",
    "I need to escalate my issue to the Assistant Faculty Officer",
    "Contact information for work integrated learning office",
    "Who is in charge of career guidance and job placement?",
    "Give me the contact details for the university cashiers desk",
    "Who do I contact about remarking application inquiries?",
    "Phone extension for the graduation office",
    "How do I reach the sports and culture department?",
    "Email address for academic appeals committee secretary",
    "I want to speak with someone from student affairs",
    "Provide phone numbers for all regional campus coordinators",
    "Escalate this request to human support"
]

# 5. CLARIFICATION_RESPONSE (50 queries)
CLARIFICATION_QUERIES = [
    "Semester 1",
    "Semester 2",
    "Full-time",
    "Part-time",
    "Distance education",
    "Bachelor of Computer Science",
    "Undergraduate",
    "Postgraduate",
    "Windhoek main campus",
    "First opportunity",
    "Second opportunity",
    "Yes, semester 1 please",
    "Computer Science",
    "Civil Engineering",
    "Bachelor of Accounting",
    "Master of Science in Information Technology",
    "No, full-time mode",
    "NSSCO level",
    "NSSCAS level",
    "Faculty of Computing and Informatics",
    "Faculty of Engineering",
    "2026 academic year",
    "June exam session",
    "November exams",
    "DSA611S",
    "PRG510S",
    "OOP611S",
    "Yes",
    "No",
    "Main campus",
    "Lüderitz campus",
    "Rundu centre",
    "Katima Mulilo",
    "Honours degree",
    "Diploma program",
    "PhD in Informatics",
    "I am an international student",
    "Namibian citizen",
    "Yes that is correct",
    "First year",
    "Final year student",
    "Second year",
    "Third year",
    "Semester 1 only",
    "Distance mode via COLL",
    "I want the morning session",
    "The 08H00 exam slot",
    "Engineering mathematics",
    "Commerce faculty",
    "Health sciences"
]

# 6. OUT_OF_SCOPE (60 queries)
OUT_OF_SCOPE_QUERIES = [
    "What is the weather like in Windhoek today?",
    "Write me a poem about machine learning and python",
    "Who is your favorite football player?",
    "Tell me a funny joke",
    "How do I bake chocolate chip cookies from scratch?",
    "Who was the first president of the United States?",
    "Can you do my history homework for me?",
    "What is the capital city of Australia?",
    "How much does tuition cost at UNAM University of Namibia?",
    "Are aliens real?",
    "Recommend a good sci-fi movie to watch tonight",
    "What is the meaning of life?",
    "Can you play chess with me?",
    "Who won the UEFA Champions League last season?",
    "How do I fix a leaking water pipe?",
    "Write a rap song about studying for exams",
    "What is the stock price of Apple right now?",
    "Translate this sentence to French for me",
    "Tell me the latest political news in Namibia",
    "Where is the nearest pizza place in town?",
    "Can you generate an image of a cybernetic cat?",
    "How tall is Mount Kilimanjaro?",
    "What is the formula for Einstein's theory of relativity?",
    "Do you believe in ghosts?",
    "Can you give me medical diagnosis for my headache?",
    "How do I hack a Wi-Fi password?",
    "Who is the CEO of Google?",
    "What are the best tourist places to visit in Swakopmund?",
    "Can you write an essay about climate change for high school?",
    "What time is it in Tokyo right now?",
    "Who wrote Hamlet?",
    "How many countries are there in Africa?",
    "Teach me how to play guitar chords",
    "What is the best smartphone to buy under 5000 dollars?",
    "Write a short story about a dragon in the desert",
    "How do I make a paper airplane fly far?",
    "Can you teach me conversational German?",
    "Who won the Rugby World Cup?",
    "What is the currency of Japan?",
    "How fast does light travel in vacuum?",
    "Tell me horoscope for Pisces today",
    "How do I change a flat car tire?",
    "What is the highest mountain in Namibia?",
    "Recommend a book on psychology",
    "Can you write a resume for a marketing job in London?",
    "What is the recipe for traditional Namibian potjiekos?",
    "How do I download music from YouTube?",
    "Who invented the telephone?",
    "Can you solve this riddle for me?",
    "What are the symptoms of common cold?",
    "How do airplanes stay in the air?",
    "Write an apology message to my friend",
    "What is the deepest ocean trench?",
    "How do solar panels work?",
    "Who painted the Mona Lisa?",
    "Tell me an interesting fun fact about dolphins",
    "Can you help me design a fashion outfit?",
    "What is the distance from Earth to Mars?",
    "How do I learn how to swim?",
    "What is the speed of sound?"
]

DATASET_CONFIG = {
    "check_calendar_date": CALENDAR_QUERIES,
    "check_module_prereq": PREREQ_QUERIES,
    "query_general_policy": POLICY_QUERIES,
    "escalate_human_contact": CONTACT_QUERIES,
    "clarification_response": CLARIFICATION_QUERIES,
    "out_of_scope": OUT_OF_SCOPE_QUERIES
}


SPLIT_ALLOCATION = {
    "check_calendar_date": (49, 10, 11),     # 70 total
    "check_module_prereq": (53, 11, 11),     # 75 total
    "query_general_policy": (63, 14, 13),    # 90 total
    "escalate_human_contact": (39, 8, 8),    # 55 total
    "clarification_response": (34, 8, 8),    # 50 total
    "out_of_scope": (42, 9, 9)               # 60 total
}


def build_dataset() -> List[Dict]:
    random.seed(42)
    dataset = []
    uid = 1

    for intent, queries in DATASET_CONFIG.items():
        shuffled = list(queries)
        random.shuffle(shuffled)
        n_train, n_val, n_test = SPLIT_ALLOCATION[intent]

        for idx, text in enumerate(shuffled):
            if idx < n_train:
                split = "train"
            elif idx < n_train + n_val:
                split = "val"
            else:
                split = "test"

            dataset.append({
                "id": f"UTT_{uid:04d}",
                "text": text,
                "intent": intent,
                "split": split
            })
            uid += 1

    # Final shuffle across intents to mix records
    random.shuffle(dataset)
    return dataset


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dataset = build_dataset()

    train_data = [d for d in dataset if d["split"] == "train"]
    val_data = [d for d in dataset if d["split"] == "val"]
    test_data = [d for d in dataset if d["split"] == "test"]

    # 1. Save unified dataset
    unified = {
        "metadata": {
            "total_samples": len(dataset),
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "test_samples": len(test_data),
            "intents": list(DATASET_CONFIG.keys()),
            "split_ratio": {"train": 0.70, "val": 0.15, "test": 0.15},
            "random_seed": 42
        },
        "data": dataset
    }

    unified_path = os.path.join(OUTPUT_DIR, "intent_dataset.json")
    with open(unified_path, "w", encoding="utf-8") as f:
        json.dump(unified, f, indent=2, ensure_ascii=False)

    # 2. Save individual splits for downstream modeling
    with open(os.path.join(OUTPUT_DIR, "train.json"), "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    with open(os.path.join(OUTPUT_DIR, "val.json"), "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)

    with open(os.path.join(OUTPUT_DIR, "test.json"), "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("INTENT CLASSIFICATION DATASET GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total Samples : {len(dataset)}")
    print(f"Train Split   : {len(train_data):3d} samples ({(len(train_data)/len(dataset))*100:.1f}%)")
    print(f"Val Split     : {len(val_data):3d} samples ({(len(val_data)/len(dataset))*100:.1f}%)")
    print(f"Test Split    : {len(test_data):3d} samples ({(len(test_data)/len(dataset))*100:.1f}%)")
    print("\nClass Distribution per Split:")
    print(f"{'Intent':25} | {'Total':5} | {'Train':5} | {'Val':4} | {'Test':4}")
    print("-" * 55)
    for intent in DATASET_CONFIG.keys():
        tot = sum(1 for d in dataset if d["intent"] == intent)
        tr = sum(1 for d in train_data if d["intent"] == intent)
        va = sum(1 for d in val_data if d["intent"] == intent)
        te = sum(1 for d in test_data if d["intent"] == intent)
        print(f"{intent:25} | {tot:5d} | {tr:5d} | {va:4d} | {te:4d}")
    print("=" * 60)
    print(f"[*] Datasets written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
