from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_cors import CORS
from dotenv import load_dotenv
import os
from openai import OpenAI
import traceback
import logging
import requests
from bs4 import BeautifulSoup

load_dotenv()

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)
CORS(app)

logging.basicConfig(level=logging.DEBUG)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Project(db.Model):
    __tablename__ = 'projects'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    website_url = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    updated_at = db.Column(db.DateTime, default=db.func.current_timestamp(), onupdate=db.func.current_timestamp())

class Step(db.Model):
    __tablename__ = 'steps'
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'))
    step_number = db.Column(db.Integer, nullable=False)
    title = db.Column(db.String(255), nullable=False)
    content = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    updated_at = db.Column(db.DateTime, default=db.func.current_timestamp(), onupdate=db.func.current_timestamp())

class WebsiteSummary(db.Model):
    __tablename__ = 'website_summaries'
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'), unique=True)
    content = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    updated_at = db.Column(db.DateTime, default=db.func.current_timestamp(), onupdate=db.func.current_timestamp())

class Context(db.Model):
    __tablename__ = 'contexts'
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'))
    step_number = db.Column(db.Integer, nullable=False)
    content = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    updated_at = db.Column(db.DateTime, default=db.func.current_timestamp(), onupdate=db.func.current_timestamp())

def fetch_website_data(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        title = soup.title.string if soup.title else "No title found"
        meta_description = soup.find('meta', attrs={'name': 'description'})
        meta_description = meta_description['content'] if meta_description else "No meta description found"
        
        h1_tags = [h1.text for h1 in soup.find_all('h1')]
        h2_tags = [h2.text for h2 in soup.find_all('h2')]
        
        ga_tag = "Google Analytics tag found" if "google-analytics.com" in response.text else "No Google Analytics tag found"
        fb_pixel = "Facebook Pixel found" if "connect.facebook.net" in response.text else "No Facebook Pixel found"
        
        return {
            "url": url,
            "title": title,
            "meta_description": meta_description,
            "h1_tags": h1_tags,
            "h2_tags": h2_tags,
            "ga_tag": ga_tag,
            "fb_pixel": fb_pixel,
            "full_html": response.text[:5000]  # First 5000 characters of HTML
        }
    except Exception as e:
        logging.error(f"Error fetching website data: {str(e)}")
        return {"error": str(e)}

def generate_prompt(step, context):
    prompts = {
        1: """Perform an audit on the following website, focusing on:
        1. On-page SEO opportunities
        2. Tracking tag installation (GA and FB tags)
        3. Overall UI/UX opportunities
        4. CRO opportunities
        
        Additionally, provide a brief summary of the website's content and purpose.
        
        Here's the website data:
        URL: {url}
        Title: {title}
        Meta Description: {meta_description}
        H1 Tags: {h1_tags}
        H2 Tags: {h2_tags}
        Google Analytics: {ga_tag}
        Facebook Pixel: {fb_pixel}
        
        Additional HTML content:
        {full_html}
        
        Please structure your response as follows:
        1. Website Summary: [Brief summary of the website's content and purpose]
        2. SEO Audit: [Your SEO audit findings]
        3. Tracking Tags: [Your findings on tracking tags]
        4. UI/UX Opportunities: [Your UI/UX observations]
        5. CRO Opportunities: [Your CRO suggestions]""",
        2: """Based on the following website summary and audit results, make intelligent guesses on the target client avatar(s) for this business. Give a detailed avatar including demographics, psychographics, pain points, fears, ideal outcomes as it pertains to this business for each of the target avatars.

        Website Summary: {website_summary}
        Audit Results: {audit_results}""",
        3: "Create a list of unique value propositions for the business based on the following information. These value props should be incredibly niche-focused and should be formatted as either: 1. 'We help {{market}} do {{thing}} without {{pain point}}.' 2. '{{service or product}} for {{descriptor}} {{client avatar}}.' 3. 'We help {{avatar}} do {{thing}} so they can {{ideal outcome}}.' Information: {previous_steps}",
        4: """Perform keyword research based on the context from previous steps. Focus on determining:
        1. Main target keyword with a monthly search volume greater than 500, keyword difficulty below 21 and more than 40 related keywords
        2. Results should include the target keyword, the search volume, keyword difficulty and a list of related keywords to help rank for that target keyword.
        
        Previous context:
        {previous_steps}
        
        Format your response as follows:
        Target Keyword: [keyword]
        Search Volume: [volume]
        Keyword Difficulty: [difficulty]
        Related Keywords: [keyword1], [keyword2], [keyword3], ...""",
        5: """Create a detailed content marketing plan based on a hub and spoke or content silo/cluster approach using the target keyword and related keywords from the previous step. The entire content plan should be based on ranking for the target keyword.

        Keyword research results:
        {keyword_research}

        Previous context:
        {previous_steps}""",
        6: "Based on context from all previous steps, create a list of lead magnets and free tool marketing options to reach our target avatar with the highest purchase intent. Context: {previous_steps}",
        7: "Create a detailed email marketing plan that includes a nurture sequence, sales sequence and ongoing content via email to keep the mailing list warm. Use the following context: {previous_steps}",
        8: "Create a plan for content and emails that address all other stages of awareness. Use the following context: {previous_steps}",
        9: "Create a point-by-point report detailing the entire plan that includes everything from the above steps. This plan should break all learnings down into multiple projects that can be completed separately from each other. The projects should be ordered by lowest-effort, highest return to highest-effort, lowest return so we're focusing on knocking out the easy but impactful projects early, then building out the longer term, high-value projects over time. Context: {previous_steps}",
        10: "Give me a list of ideas for other income streams this business could create related to everything we've discussed so far. Each income stream should include a summary of the business, how it would be monetized and any further details necessary to determine if it's something the company wants to pursue. These income streams should include (but not be limited to) online courses, digital downloads, productized services, paid content (newsletters, etc.), monetized directories, memberships and SaaS opportunities. Context: {previous_steps}"
    }
    return prompts.get(step, "").format(**context)

@app.route('/api/projects', methods=['GET'])
def get_projects():
    try:
        projects = Project.query.all()
        return jsonify([
            {'id': project.id, 'name': project.name, 'website_url': project.website_url}
            for project in projects
        ])
    except Exception as e:
        logging.error(f"Error in get_projects: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/projects', methods=['POST'])
def create_project():
    try:
        data = request.json
        new_project = Project(name=data['name'], website_url=data['website_url'])
        db.session.add(new_project)
        db.session.commit()
        return jsonify({'id': new_project.id, 'name': new_project.name, 'website_url': new_project.website_url}), 201
    except Exception as e:
        logging.error(f"Error in create_project: {str(e)}")
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@app.route('/api/process', methods=['POST'])
def process_step():
    logging.info("Received request to /api/process")
    try:
        data = request.json
        logging.info(f"Received data: {data}")
        
        step = data['step']
        project_id = data['project_id']
        context = data.get('context', {})
        
        project = Project.query.get(project_id)
        if not project:
            return jsonify({"error": "Project not found"}), 404

        if step == 1:
            website_data = fetch_website_data(project.website_url)
            context.update(website_data)
        elif step == 2:
            previous_step = Step.query.filter_by(project_id=project_id, step_number=1).first()
            context['audit_results'] = previous_step.content if previous_step else ''
        elif step == 5:
            previous_step = Step.query.filter_by(project_id=project_id, step_number=4).first()
            context['keyword_research'] = previous_step.content if previous_step else ''
        
        prompt = generate_prompt(step, context)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a marketing strategy AI assistant with expertise in website auditing, optimization, and keyword research."},
                {"role": "user", "content": prompt}
            ]
        )
        
        result = response.choices[0].message.content
        
        new_step = Step(project_id=project_id, step_number=step, title=f"Step {step}", content=result)
        db.session.add(new_step)
        
        if step == 1:
            summary_start = result.find("Website Summary:") + len("Website Summary:")
            summary_end = result.find("2. SEO Audit:")
            website_summary = result[summary_start:summary_end].strip()
            new_summary = WebsiteSummary(project_id=project_id, content=website_summary)
            db.session.add(new_summary)
        
        new_context = Context(project_id=project_id, step_number=step, content=context)
        db.session.add(new_context)
        
        db.session.commit()
        
        return jsonify({"response": result, "context": context})
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        logging.error(traceback.format_exc())
        db.session.rollback()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(debug=True)