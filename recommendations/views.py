import os
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, FileResponse, HttpResponseBadRequest
from django.conf import settings
from django.contrib import messages
from django.urls import reverse
from django.core.exceptions import ValidationError
from .forms import RecommendationForm
from crop_api import ProfessionalCropRecommender
from datetime import datetime

# Initialize recommender with API key from settings
recommender = ProfessionalCropRecommender()
recommender.load_and_merge_data()
model_info = recommender.train_model()

@login_required
def home(request):
    return render(request, 'recommendations/home.html', {
        'title': 'Crop Recommendation System'
    })

@login_required
def recommend(request):
    if request.method == 'POST':
        form = RecommendationForm(request.POST)
        if form.is_valid():
            try:
                user_input = {
                    'soil_ph': float(form.cleaned_data['soil_ph']),
                    'soil_temp': float(form.cleaned_data['soil_temp']),
                    'soil_type': form.cleaned_data['soil_type'],
                    'rainfall': float(form.cleaned_data['rainfall']),
                    'humidity': float(form.cleaned_data['humidity'])
                }
                location = form.cleaned_data.get('location', '').strip()

                recommender.validate_inputs(**user_input)
                suggestions = recommender.generate_suggestions(user_input, location)
                
                if not suggestions:
                    messages.warning(request, "No crops matched your conditions. Try adjusting your parameters.")
                    return redirect(reverse('recommendations:recommend'))
                
                report = recommender.generate_report(suggestions, model_info, user_input, location)
                txt_file = recommender.save_report_to_file(report)
                html_file = recommender.save_report_as_html(report)
                plot_path = recommender.plot_recommendations(suggestions)

                if not all(os.path.exists(f) for f in [txt_file, html_file, plot_path]):
                    raise FileNotFoundError("Could not generate all report files")

                request.session['report_files'] = {
                    'text': txt_file,
                    'html': html_file,
                    'plot': plot_path
                }

                context = {
                    'report': report,
                    'suggestions': suggestions,
                    'plot_path': os.path.relpath(plot_path, os.path.join(settings.BASE_DIR, 'static')),
                    'user_input': user_input,
                    'location': location if location else 'your area'
                }
                return render(request, 'recommendations/results.html', context)

            except ValueError as e:
                messages.error(request, f"Invalid input value: {str(e)}")
            except ValidationError as e:
                messages.error(request, f"Validation error: {str(e)}")
            except Exception as e:
                messages.error(request, "An error occurred while processing your request")
                print(f"Recommendation Error: {str(e)}")
            return redirect(reverse('recommendations:recommend'))
    else:
        form = RecommendationForm()

    return render(request, 'recommendations/recommend.html', {
        'form': form,
        'title': 'Get Crop Recommendations'
    })

@login_required
def download_report(request, file_type):
    if file_type not in {'text', 'html', 'plot'}:
        return HttpResponseBadRequest("Invalid file type requested")

    file_path = request.session.get('report_files', {}).get(file_type)
    
    if not file_path:
        messages.error(request, "No report generated yet")
        return redirect(reverse('recommendations:recommend'))
    
    if not os.path.exists(file_path):
        messages.error(request, "Report file has expired or was deleted")
        return redirect(reverse('recommendations:recommend'))

    try:
        extension = 'png' if file_type == 'plot' else file_type
        filename = f"crop_recommendation_{request.user.username}_{file_type}.{extension}"
        
        response = FileResponse(
            open(file_path, 'rb'),
            as_attachment=True,
            filename=filename,
            content_type='text/plain' if file_type == 'text' else 
                        'text/html' if file_type == 'html' else 
                        'image/png'
        )
        return response
    except Exception as e:
        messages.error(request, f"Could not prepare download: {str(e)}")
        return redirect(reverse('recommendations:results'))

@login_required
def chat_api(request):
    if request.method != 'POST' or not request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return JsonResponse({'error': 'Invalid request'}, status=400)

    question = request.POST.get('question', '').strip()
    if not question:
        return JsonResponse({'error': 'Empty question'}, status=400)

    try:
        response = recommender.chat_with_assistant(question)
        
        # Store the conversation in session if you want to maintain history
        chat_history = request.session.get('chat_history', [])
        chat_history.append({
            'question': question,
            'response': response,
            'timestamp': str(datetime.now())
        })
        request.session['chat_history'] = chat_history
        
        return JsonResponse({
            'response': response,
            'status': 'success'
        })
    except Exception as e:
        return JsonResponse({
            'error': 'Failed to process your question',
            'details': str(e)
        }, status=500)

@login_required
def download_chat(request):
    """Download the entire chat history as a text file"""
    chat_history = request.session.get('chat_history', [])
    if not chat_history:
        messages.warning(request, "No chat history available to download")
        return redirect(reverse('recommendations:home'))

    # Format the chat history for download
    chat_text = "AGRICULTURE ASSISTANT CHAT HISTORY\n\n"
    chat_text += f"User: {request.user.username}\n"
    chat_text += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    for entry in chat_history:
        chat_text += f"Q: {entry['question']}\n"
        chat_text += f"A: {entry['response']}\n"
        chat_text += f"Timestamp: {entry['timestamp']}\n\n"
        chat_text += "="*50 + "\n\n"

    # Create a temporary file
    filename = f"agriculture_chat_{request.user.username}_{datetime.now().strftime('%Y%m%d')}.txt"
    temp_file = os.path.join(settings.MEDIA_ROOT, 'temp_chats', filename)
    
    os.makedirs(os.path.dirname(temp_file), exist_ok=True)
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(chat_text)

    # Serve the file for download
    response = FileResponse(open(temp_file, 'rb'))
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    return response