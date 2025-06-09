
from fpdf import FPDF
import os
from pathlib import Path

def create_job_template(job_title, job_content, output_filename):
    """Create a PDF job template with formatted content"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Add a title
    pdf.set_font("Arial", 'B', size=16)
    pdf.cell(200, 10, txt=job_title, ln=True, align='C')
    pdf.ln(5)
    
    # Add content
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, job_content)
    
    # Save the PDF
    pdf_path = Path(output_filename)
    pdf.output(str(pdf_path))
    print(f"Created job template: {pdf_path}")
    return pdf_path

def main():
    """Create sample job templates for common roles"""
    # Ensure we're saving to the correct directory
    script_dir = Path(__file__).parent
    
    # Software Engineer template
    software_engineer = """
1. Education and Experience:
   - Bachelor's degree in Computer Science, Software Engineering, or related field.
   - 3-5 years of professional software development experience.

2. Technical Skills:
   - Proficiency in at least one programming language such as Python, Java, C#, or JavaScript.
   - Experience with front-end and back-end development.
   - Knowledge of software development methodologies (Agile, Scrum, etc.).
   - Familiarity with database design and SQL.
   - Experience with version control systems (Git).
   - Understanding of cloud platforms (AWS, Azure, or Google Cloud).

3. Soft Skills:
   - Strong problem-solving abilities and analytical thinking.
   - Excellent communication and collaboration skills.
   - Ability to work in a team environment.
   - Self-motivated and proactive approach to work.
   - Attention to detail and commitment to code quality.

4. Responsibilities:
   - Design, develop, and maintain software applications.
   - Write clean, efficient, and well-documented code.
   - Collaborate with cross-functional teams to define requirements.
   - Troubleshoot, debug, and optimize application performance.
   - Participate in code reviews and ensure code quality.
   - Contribute to technical documentation.
"""

    # Data Scientist template
    data_scientist = """
1. Education and Experience:
   - Master's or Ph.D. in Computer Science, Statistics, Mathematics, or related field.
   - 2-4 years of experience in data science or related field.

2. Technical Skills:
   - Strong proficiency in Python, R, or similar programming languages.
   - Experience with data manipulation and analysis libraries (Pandas, NumPy, etc.).
   - Knowledge of machine learning frameworks (Scikit-learn, TensorFlow, PyTorch).
   - Experience with data visualization tools (Matplotlib, Seaborn, Tableau).
   - Understanding of statistical analysis and modeling techniques.
   - SQL database knowledge and data extraction skills.

3. Soft Skills:
   - Excellent analytical and critical thinking abilities.
   - Strong communication skills to present findings to non-technical stakeholders.
   - Curiosity and eagerness to learn new technologies and methods.
   - Problem-solving mindset and attention to detail.
   - Ability to work independently and as part of a team.

4. Responsibilities:
   - Collect, process, and analyze large datasets to extract insights.
   - Develop machine learning models to address business problems.
   - Present findings and recommendations to stakeholders.
   - Collaborate with engineering teams to implement data science solutions.
   - Research and implement new methodologies to improve existing models.
   - Create visualizations and reports to communicate results effectively.
"""

    # Project Manager template
    project_manager = """
1. Education and Experience:
   - Bachelor's degree in Business, Computer Science, or related field.
   - 5+ years of experience in project management.
   - PMP certification preferred.

2. Technical Skills:
   - Proficiency in project management software (JIRA, Microsoft Project, etc.).
   - Knowledge of project management methodologies (Agile, Waterfall, etc.).
   - Understanding of software development lifecycle.
   - Experience with budgeting and resource allocation.
   - Data analysis and reporting skills.

3. Soft Skills:
   - Exceptional leadership and team management abilities.
   - Strong communication and interpersonal skills.
   - Problem-solving and conflict resolution capabilities.
   - Organizational skills and attention to detail.
   - Ability to work under pressure and meet deadlines.

4. Responsibilities:
   - Lead and manage projects throughout their lifecycle.
   - Define project scope, goals, and deliverables with stakeholders.
   - Develop project plans, timelines, and budgets.
   - Allocate resources and coordinate team members.
   - Track project progress and address any issues or bottlenecks.
   - Communicate project status to all stakeholders.
   - Identify and mitigate project risks.
   - Ensure projects are completed on time, within scope, and budget.
"""

    # Create the templates
    templates = [
        ("Software Engineer Job Requirements", software_engineer, "Software_Engineer_Template.pdf"),
        ("Data Scientist Job Requirements", data_scientist, "Data_Scientist_Template.pdf"),
        ("Project Manager Job Requirements", project_manager, "Project_Manager_Template.pdf")
    ]
    
    for title, content, filename in templates:
        output_path = script_dir / filename
        create_job_template(title, content, output_path)

if __name__ == "__main__":
    main()
