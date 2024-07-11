from flask import Flask, request, jsonify
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
CORS(app)

logging.basicConfig(level=logging.DEBUG)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
        
        Please provide a detailed analysis and specific recommendations for improvement in each area.""",
        2: "Based on the following audit results, make intelligent guesses on the target client avatar(s) for this business. Give a detailed avatar including demographics, psychographics, pain points, fears, ideal outcomes as it pertains to this business for each of the target avatars. Audit results: {audit_results}",
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


@app.route('/api/process', methods=['POST'])
def process_step():
    logging.info("Received request to /api/process")
    try:
        data = request.json
        logging.info(f"Received data: {data}")
        
        step = data['step']
        context = data.get('context', {})
        
        if step == 1:
            website_data = fetch_website_data(context['website'])
            context.update(website_data)
        elif step == 2:
            context['audit_results'] = context.get('step1_results', '')
        elif step == 5:
            context['keyword_research'] = context.get('step4_results', '')
        
        context['previous_steps'] = "\n".join([context.get(f'step{i}_results', '') for i in range(1, step)])
        
        logging.info(f"Processing step {step} with context: {context}")
        
        prompt = generate_prompt(step, context)
        
        logging.info(f"Generated prompt: {prompt}")
        
        if not client.api_key:
            raise ValueError("OpenAI API key is not set")
        
        logging.info("Sending request to OpenAI API")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a marketing strategy AI assistant with expertise in website auditing, optimization, and keyword research."},
                {"role": "user", "content": prompt}
            ]
        )
        
        result = response.choices[0].message.content
        
        logging.info(f"Received response from OpenAI API: {result[:100]}...")  # Log first 100 chars of response
        
        return jsonify({"response": result})
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(debug=True)