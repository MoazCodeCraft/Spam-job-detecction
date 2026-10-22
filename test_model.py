
import joblib

# Load the saved model and TF-IDF vectorizer
saved_data = joblib.load("saved_model/spam_job_model.pkl")

model = saved_data["model"]
tfidf = saved_data["tfidf"]

# Example job posting
job_title = "IC&E Technician"
company_profile = """
<p>Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â <img src="#URL_a9fa980073b093916a3b892dbcc0c2f8ad7ce8a6ee981f60f1c9b0f8108746d6#"></p>
<p><b>Staffing &amp; Recruiting done right for the Oil &amp; Energy Industry!</b></p>
<p><b>Represented candidates are automatically granted the following perks: </b></p>
<ul>
<li><b>Expert negotiations on your behalf, maximizing your compensation package and implimenting ongoing increasesÂ </b></li>
<li><b>Significant signing bonus by Refined Resources (in addition to any potential signing bonuses our client companies offer)</b></li>
<li><b>1 Year access to AnyPerk: significant corporate discounts on cell phones, event tickets, house cleaning and everything inbetween. Â You'll save thousands on daily expendituresÂ </b></li>
<li><b>Professional Relocation Services for out of town candidates</b></li>
</ul><p><b>* All candidates are encouraged to participate in our<i> Referral Bonus Program</i> ranging anywhere from <i>$500 - $1,000</i> for all successfully hired candidates... referred directly to the Refined Resources team</b></p>
<p><b>Please submit referrals via online <a href="#URL_0ceeb8a5904676d1a850bdf02e3e6d8e2e4ca8a086bca71e82175a42e4d7be23#?f=4GrlL4b1ZzoSHOtZ5yVbSw" rel="nofollow" class="external">R</a><a href="#URL_0ceeb8a5904676d1a850bdf02e3e6d8e2e4ca8a086bca71e82175a42e4d7be23#?f=4GrlL4b1ZzoSHOtZ5yVbSw" rel="nofollow" class="external">eferral Form</a></b></p>
<p></p>
<p><b>Thank you and we look forward to working with you soon! Â </b></p>
<p><a href="https://s3.#URL_921b06bd19182411a4b5c452cf459eb5bbed775def9a0ce28c6396708ac09772#" rel="nofollow" class="external"><b><img src="https://s3.#URL_921b06bd19182411a4b5c452cf459eb5bbed775def9a0ce28c6396708ac09772#"></b></a></p>
<p></p>
<h3>[ Click to enlarge Image ]</h3>"""
description = """
<p><b><img src="#URL_ae07dc35dfe86ebc1101b48eec77574871f2ba0daaf971ad3f2e2246bee5b31d#"></b></p>
<p></p>
<p></p>
<p><b>IC&amp;E Technician | Bakersfield, CA Mt. Poso</b></p>
<p></p>
<p><b>Principal Duties and Responsibilities:Â </b></p>
<ul>
<li>Calibrates, tests, maintains, troubleshoots, and installs all power plant instrumentation, control systems and electrical equipment.</li>
<li>Performs maintenance on motor control centers, motor operated valves, generators, excitation equipment and motors.</li>
<li>Performs preventive, predictive and corrective maintenance on equipment, coordinating work with various team members.</li>
<li>Designs and installs new equipment and/or system modifications.</li>
<li>Troubleshoots and performs maintenance on DC backup power equipment, process controls, programmable logic controls (PLC), and emission monitoring equipment.</li>
<li>Uses maintenance reporting system to record time and material use, problem identified and corrected, and further action required; provides complete history of maintenance on equipment.</li>
<li>Schedule, coordinate, work with and monitor contractors on specific tasks, as required.</li>
<li>Follows safe working practices at all times.</li>
<li>Identifies safety hazards and recommends solutions.</li>
<li>Follows environmental compliance work practices.</li>
<li>Identifies environmental non-compliance problems and assist in implementing solutions.</li>
<li>Assists other team members and works with all departments to support generating station in achieving their performance goals.</li>
<li>Trains other team members in the areas of instrumentation, control, and electrical systems.</li>
<li>Performs housekeeping assignments, as directed.</li>
<li>Conduct equipment and system tagging according to company and plant rules and regulations.</li>
<li>Perform equipment safety inspections, as required, and record results as appropriate.Â </li>
<li>Participate in small construction projects.Â  Read and interpret drawings, sketches, prints, and specifications, as required.</li>
<li>Orders parts as needed to affect maintenance and repair.</li>
<li>Performs Operations tasks on an as-needed basis and other tasks as assigned.</li>
<li>Available within a reasonable response time for emergency call-ins and overtime, plus provide acceptable off-hour contact by phone and company pager.</li>
</ul><p>Â Â Â Â Â Â Â Â Â  Excellent Verbal and Written Communications Skills:</p>
<ul>
<li>Ability to coordinate work activities with other team members on technical subjects across job families.</li>
<li>Ability to work weekends, holidays, and rotating shifts, as required.</li>
</ul>
"""
requirements = """
<h3><b>Qualifications</b></h3>
<p><b>Knowledge, Skills &amp; Abilities:</b></p>
<p>Â A high school diploma or GED is required. Must have a valid driverâ€™s license. Ability to read, write, and communicate effectively in English.Â Â Good math skills.Â Four years of experience as an I&amp;C Technician and/or Electrician in a power plant environment, preferably with a strong electrical background, up to and including, voltages to 15 KV to provide the following:</p>
<ul>
<li>Demonstrated knowledge of electrical equipment, electronics, schematics, basics of chemistry and physics and controls and instrumentation.</li>
<li>Demonstrated knowledge of safe work practices associated with a power plant environment.</li>
<li>Demonstrated ability to calibrate I&amp;C systems and equipment, including analytic equipment.</li>
<li>Demonstrated ability to configure and operate various test instruments and equipment, as necessary, to troubleshoot and repair plant equipment including, but not limited to, distributed control systems, programmable logic controllers, motor control centers, transformers, generators, and continuous emissions monitor (CEM) systems.</li>
<li>Demonstrated ability to work with others in a team environment.Â </li>
</ul>"""
benefits = """
<p><b>BENEFITS</b></p>
<p><b>What is offered:</b></p>
<ul>
<li>Competitive compensation package</li>
<li>100% matched retirement fund</li>
<li>Annual vacations paid for by company</li>
<li>Significant bonus structure</li>
<li>Opportunity for advancementÂ </li>
<li>Full benefits package</li>
<li>Annual performance reviews and base salary increases</li>
<li>Annual cost of living increasesÂ </li>
<li>Sound, clean, safe and enjoyable working environment &amp; Company Culture</li>
<li>World renound management and executive team who promote from within, leverage careers and invest in employees for the long-term success of their careers and overall company/employee goals</li>
</ul><h3>
<b></b><b>Qualified candidates contact:Â </b><b>Darren Lawson | VP of Recruiting |Â <a href="mailto:#EMAIL_395225df8eed70288fc67310349d63d49d5f2ca6bc14dbb5dcbf9296069ad88c#" target="_blank">#EMAIL_395225df8eed70288fc67310349d63d49d5f2ca6bc14dbb5dcbf9296069ad88c#</a>Â | #PHONE_70128aad0c118273b0c2198a08d528591b932924e165b6a8d1272a6f9e2763d1#</b>
</h3>
<p></p>
<p></p>
<p>Â Â </p>
<p>Â </p>
<p><img src="#URL_4f324c74032b67d1b67e856a4482eaab6d2df75979ba5cd916820d3359ebe0d3#?module=AppInert&amp;action=download&amp;id=13&amp;hashed=-263076167&amp;a=1#PHONE_d101f0dd1da019c587181a70762459fdd9e2534a5c7c2a618a9b01557176fe36#"></p>"""

# Combine text using the same fields as training
job_text = " ".join([
    job_title,
    company_profile,
    description,
    requirements,
    benefits
])

# Convert text and predict
job_features = tfidf.transform([job_text])

prediction = model.predict(job_features)[0]
probabilities = model.predict_proba(job_features)[0]

fraud_probability = probabilities[
    list(model.classes_).index(1)
]

print("\nPrediction result")

if prediction == 1:
    print("Potentially fraudulent job posting")
else:
    print("Likely legitimate job posting")

print(f"Estimated fraud probability: {fraud_probability:.2%}")
