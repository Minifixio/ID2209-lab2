import gradio as gr
from llama_cpp import Llama

# Load the pretrained LLM
llm = Llama.from_pretrained(
    repo_id="minifixio/lora_model_gguf",
    filename="*Q8_0.gguf",
    verbose=False,
    chat_format="llama-3",
)

# System message for the LLM
system_message = "You are an assistant specialized in writing professional and compelling cover letters for job applications."

# Function to generate the cover letter
def generate_cover_letter(name, job_title, company_name, skills, reasons):
    # Prompt template
    prompt = f"""
    Write a professional and compelling cover letter for a job application.
    Applicant Name: {name}
    Job Title: {job_title}
    Company Name: {company_name}
    Skills: {skills}
    Reasons for applying: {reasons}
    Return only the cover letter, nothing else.
    """
    
    # Generate response
    messages = [{"role": "system", "content": system_message},
                {"role": "user", "content": prompt}]
    response = ""
    for message in llm.create_chat_completion(
        messages,
        max_tokens=1024,
        stream=True,
        temperature=0.7,
        top_p=0.95,
    ):
        delta = message["choices"][0]["delta"]
        if "content" not in delta:
            continue
        response += delta["content"]
    return response

# Gradio Interface
with gr.Blocks() as interface:
    gr.Markdown("# Cover Letter Generator")
    gr.Markdown("Use this app to generate professional cover letters by providing the necessary details.")
    
    with gr.Row():
        name_input = gr.Textbox(label="Your Name", placeholder="Enter your name here")
        job_title_input = gr.Textbox(label="Job Title", placeholder="Enter the job title here")
        company_name_input = gr.Textbox(label="Company Name", placeholder="Enter the company name here")
    
    skills_input = gr.Textbox(label="Your Skills", placeholder="List your skills separated by commas")
    reasons_input = gr.Textbox(label="Why You're Applying", placeholder="Explain your reasons for applying")
    
    generate_button = gr.Button("Generate Cover Letter")
    output_box = gr.Textbox(label="Generated Cover Letter", placeholder="Your cover letter will appear here", lines=10)

    # Connect the generate button with the function
    generate_button.click(
        fn=generate_cover_letter,
        inputs=[name_input, job_title_input, company_name_input, skills_input, reasons_input],
        outputs=output_box
    )

# Launch the app
if __name__ == "__main__":
    interface.launch(debug=True)
