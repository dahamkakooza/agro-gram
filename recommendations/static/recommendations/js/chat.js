document.addEventListener('DOMContentLoaded', function() {
    const chatForm = document.getElementById('chatForm');
    const questionInput = document.getElementById('questionInput');
    const chatContainer = document.getElementById('chatContainer');
    
    chatForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const question = questionInput.value.trim();
        if (!question) return;
        
        // Add user message to chat
        const userMessage = document.createElement('div');
        userMessage.className = 'message user-message';
        userMessage.textContent = question;
        chatContainer.appendChild(userMessage);
        
        // Clear input
        questionInput.value = '';
        
        // Scroll to bottom
        chatContainer.scrollTop = chatContainer.scrollHeight;
        
        // Send to server
        fetch('{% url "chat_api" %}', {
            method: 'POST',
            headers: {
                'X-Requested-With': 'XMLHttpRequest',
                'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value,
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `question=${encodeURIComponent(question)}`
        })
        .then(response => response.json())
        .then(data => {
            const botMessage = document.createElement('div');
            botMessage.className = 'message bot-message';
            botMessage.innerHTML = data.response.replace(/\n/g, '<br>');
            chatContainer.appendChild(botMessage);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        })
        .catch(error => {
            console.error('Error:', error);
            const errorMessage = document.createElement('div');
            errorMessage.className = 'message bot-message text-danger';
            errorMessage.textContent = "Sorry, I couldn't process your request. Please try again.";
            chatContainer.appendChild(errorMessage);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        });
    });
});