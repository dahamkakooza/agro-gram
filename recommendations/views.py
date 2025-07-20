import os
import sys
import json
from pathlib import Path
from datetime import datetime
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, FileResponse, HttpResponseBadRequest
from django.conf import settings
from django.contrib import messages
from django.urls import reverse
from django.core.exceptions import ValidationError
from .forms import RecommendationForm

# Import Agro-gram.py from the specified path
agro_gram_path = r"C:\Users\HP\Desktop\crop_reccomder\Agro-gram.py"
sys.path.insert(0, str(Path(agro_gram_path).parent))

try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("agro_gram", agro_gram_path)
    agro_gram = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agro_gram)
    ProfessionalCropRecommender = agro_gram.ProfessionalCropRecommender
except Exception as e:
    raise ImportError(f"Failed to import Agro-gram.py: {str(e)}")

# Initialize recommender
recommender = ProfessionalCropRecommender()

def get_suitability_label(confidence: float) -> str:
    """Convert confidence score to human-readable suitability label"""
    if confidence >= 0.9:
        return "Excellent"
    elif confidence >= 0.7:
        return "Very Good"
    elif confidence >= 0.5:
        return "Good"
    elif confidence >= 0.3:
        return "Moderate"
    elif confidence >= 0.1:
        return "Marginal"
    else:
        return "Poor"

def prepare_recommendation_context(suggestions, user_input, location=None):
    """Prepare the context data for rendering recommendations"""
    top_recommendations = []
    for i, suggestion in enumerate(suggestions[:5]):  # Top 5 recommendations
        market_data = recommender.CROP_MARKET_DATA.get(suggestion['Crop'], {})
        
        top_recommendations.append({
            'rank': i + 1,
            'crop': suggestion['Crop'],
            'confidence': f"{suggestion['Confidence']:.1%}",
            'profit': f"${market_data.get('profit_per_acre', 0):,.0f}",
            'risk': str(suggestion.get('WeatherRisk', 'Moderate')),
            'suitability': get_suitability_label(suggestion['Confidence'])
        })
    
    return {
        'suggestions': top_recommendations,
        'user_input': user_input,
        'location': location if location else 'your area'
    }

@login_required
def home(request):
    """Home page view"""
    return render(request, 'recommendations/home.html', {
        'title': 'Crop Recommendation System'
    })

@login_required
def recommend(request):
    """Main recommendation view"""
    if request.method == 'POST':
        form = RecommendationForm(request.POST)
        if form.is_valid():
            try:
                # Prepare user input
                user_input = {
                    'soil_ph': float(form.cleaned_data['soil_ph']),
                    'soil_temp': float(form.cleaned_data['soil_temp']),
                    'soil_type': form.cleaned_data['soil_type'],
                    'rainfall': int(form.cleaned_data['rainfall']),
                    'humidity': float(form.cleaned_data['humidity'])
                }
                location = form.cleaned_data.get('location', '').strip() or None

                # Validate inputs
                recommender.validate_inputs(**user_input)
                
                # Get recommendations
                suggestions, weather_data = recommender.generate_suggestions(user_input, location)
                
                if not suggestions:
                    messages.warning(request, "No crops matched your conditions. Try adjusting your parameters.")
                    return redirect(reverse('recommendations:recommend'))
                
                # Generate report
                report = recommender.generate_report(
                    suggestions, 
                    {'best_score': 0.85, 'best_params': {}},  # Dummy model info
                    user_input, 
                    location,
                    weather_data
                )
                
                # Save outputs
                txt_file = recommender.save_report_to_file(report)
                html_file = recommender.save_report_as_html(report)
                plot_path = recommender.plot_recommendations(suggestions)

                # Store file paths in session
                request.session['report_files'] = {
                    'text': txt_file,
                    'html': html_file,
                    'plot': plot_path
                }

                # Prepare context
                context = prepare_recommendation_context(suggestions, user_input, location)
                context.update({
                    'report': report,
                    'plot_path': os.path.relpath(plot_path, os.path.join(settings.BASE_DIR, 'static')) if plot_path else None,
                })
                
                return render(request, 'recommendations/results.html', context)

            except ValueError as e:
                messages.error(request, f"Invalid input value: {str(e)}")
            except ValidationError as e:
                messages.error(request, f"Validation error: {str(e)}")
            except Exception as e:
                messages.error(request, "An error occurred while processing your request")
                print(f"Error: {str(e)}")
            return redirect(reverse('recommendations:recommend'))
    else:
        form = RecommendationForm()

    return render(request, 'recommendations/recommend.html', {
        'form': form,
        'title': 'Get Crop Recommendations'
    })

@login_required
def download_report(request, file_type):
    """Handle report downloads"""
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
    """Handle AI chat requests"""
    if request.method != 'POST' or not request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return JsonResponse({'error': 'Invalid request'}, status=400)

    question = request.POST.get('question', '').strip()
    if not question:
        return JsonResponse({'error': 'Empty question'}, status=400)

    try:
        response = recommender.chat_with_assistant(question)
        
        # Store conversation history
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
    """Download chat history"""
    chat_history = request.session.get('chat_history', [])
    if not chat_history:
        messages.warning(request, "No chat history available to download")
        return redirect(reverse('recommendations:home'))

    # Format chat history
    chat_text = "AGRICULTURE ASSISTANT CHAT HISTORY\n\n"
    chat_text += f"User: {request.user.username}\n"
    chat_text += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    for entry in chat_history:
        chat_text += f"Q: {entry['question']}\n"
        chat_text += f"A: {entry['response']}\n"
        chat_text += f"Timestamp: {entry['timestamp']}\n\n"
        chat_text += "="*50 + "\n\n"

    # Create temporary file
    filename = f"agriculture_chat_{request.user.username}_{datetime.now().strftime('%Y%m%d')}.txt"
    temp_file = os.path.join(settings.MEDIA_ROOT, 'temp_chats', filename)
    
    os.makedirs(os.path.dirname(temp_file), exist_ok=True)
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(chat_text)

    # Serve file for download
    response = FileResponse(open(temp_file, 'rb'))
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    return response