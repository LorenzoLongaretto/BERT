import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering,pipeline
from flask import Flask, render_template, request

model_dir="./modelo"
tokenizer_dir="./tokenizer"
model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)


# def predict(question_input):
#     context = "En la medicina, el corazón es un órgano muscular que bombea sangre a través de los vasos sanguíneos del sistema circulatorio. "
#     encoded_input = tokenizer(question_input, context, return_tensors='pt')
#     output = model(**encoded_input)
#     answer_start = torch.argmax(output.start_logits)
#     answer_end = torch.argmax(output.end_logits) + 1
#     answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0][answer_start:answer_end]))
#     return answer

def predict(context_input, question_input):
    encoded_input = tokenizer(question_input, context_input, return_tensors='pt')
    output = model(**encoded_input)
    answer_start = torch.argmax(output.start_logits)
    answer_end = torch.argmax(output.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0][answer_start:answer_end]))
    return answer


from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # obtener el contexto y la pregunta del formulario
        context = request.form['context_input']
        question = request.form['question_input']
        
        # cargar el modelo BERT
        salida = pipeline('question-answering', model=model, tokenizer=tokenizer)
        
        # generar una respuesta utilizando el modelo BERT
        answer = salida({'context': context, 'question': question})['answer']
        #print(answer)
        # devolver la respuesta a la plantilla HTML
        return render_template('index.html', answer=answer)
    else:
        # mostrar la plantilla HTML sin respuesta
        return render_template('index.html')
