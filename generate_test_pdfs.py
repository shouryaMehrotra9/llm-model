import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def create_pdf(filename, title, content_paragraphs):
    os.makedirs("test_cases", exist_ok=True)
    filepath = os.path.join("test_cases", filename)
    doc = SimpleDocTemplate(filepath, pagesize=letter, rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54)
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor='#1e3a8a',
        spaceAfter=15
    )
    
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=11,
        leading=16,
        spaceAfter=12
    )
    
    story = []
    # Add title
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 10))
    
    # Add content
    for text in content_paragraphs:
        story.append(Paragraph(text, body_style))
        story.append(Spacer(1, 5))
        
    doc.build(story)
    print(f"Created PDF at: {filepath}")

# Case 1: Johnson v. Smith (Breach of Contract & Verdict)
johnson_v_smith_content = [
    "CASE NO: CIV-2025-00412. IN THE SUPERIOR COURT OF CALIFORNIA, COUNTY OF SAN FRANCISCO.",
    "JOHN JOHNSON, Plaintiff, v. ALICE SMITH, Defendant.",
    "DECISION AND FINAL JUDGMENT",
    "FACTUAL BACKGROUND: On January 10, 2025, the Plaintiff, John Johnson, entered into a written agreement with the Defendant, Alice Smith, for the development and delivery of 500 units of custom database software servers. Under the terms of the contract, delivery was to be completed in full no later than June 1, 2025. The contract price was agreed to be $150,000, with the Plaintiff providing a deposit payment of $50,000 upon execution of the agreement.",
    "BREACH OF CONTRACT ANALYSIS: The Defendant, Alice Smith, failed to deliver any of the agreed-upon software servers by June 1, 2025. Despite multiple demands and warnings from the Plaintiff, the Defendant did not provide the goods, citing unexpected technical development difficulties. The court finds that the Defendant's failure to perform represents a material breach of contract, as the delivery date was a critical and essential term of the agreement.",
    "VERDICT: Upon review of the evidence, the court finds in favor of the Plaintiff, John Johnson, on the claim of breach of contract.",
    "DAMAGES AWARDED: The Plaintiff is entitled to recover damages resulting from the breach. The court hereby awards the Plaintiff damages in the total amount of $75,000. This sum represents the return of the original $50,000 deposit payment plus an additional $25,000 in consequential damages to compensate for the business losses incurred due to the operational delay.",
    "IT IS SO ORDERED. Dated: June 20, 2025."
]

# Case 2: Davis v. Mega Corp (Employment discrimination & wrongful termination)
davis_v_mega_corp_content = [
    "CASE NO: US-DIST-NY-8849. IN THE UNITED STATES DISTRICT COURT FOR THE EASTERN DISTRICT OF NEW YORK.",
    "ROBERT DAVIS, Plaintiff, v. MEGA CORP INC., Defendant.",
    "MEMORANDUM DECISION AND ORDER",
    "FACTUAL BACKGROUND: The Plaintiff, Robert Davis, was employed as a senior software engineer at Mega Corp Inc. for a period of ten consecutive years. On March 15, 2025, the Plaintiff's employment was terminated by the Defendant. The Plaintiff filed this lawsuit alleging wrongful termination, claiming the decision was discriminatory and in direct violation of state and federal employment guidelines.",
    "FINDINGS OF THE COURT: The court examined employment records and email correspondence. The evidence shows that the Plaintiff's termination did not align with Mega Corp's standard performance evaluation protocols, and was based on discriminatory motives related to age. The court determines that the termination was indeed wrongful, unlawful, and discriminatory under state guidelines.",
    "VERDICT: The court rules in favor of the Plaintiff, Robert Davis, on the wrongful termination claim.",
    "DAMAGES AWARDED: The court awards the Plaintiff a total sum of $250,000 in damages. This award is comprised of $150,000 in back pay and $100,000 in compensatory damages for emotional distress. The court denies the Plaintiff's request for punitive damages, finding insufficient evidence of malice or reckless indifference.",
    "SO ORDERED. Dated: June 25, 2025."
]

if __name__ == "__main__":
    create_pdf("johnson_v_smith.pdf", "Johnson v. Smith Final Judgment", johnson_v_smith_content)
    create_pdf("davis_v_mega_corp.pdf", "Davis v. Mega Corp Order", davis_v_mega_corp_content)
