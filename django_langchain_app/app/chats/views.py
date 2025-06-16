# views.py
from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import HttpResponseRedirect
from django import forms
from django.utils import timezone

# Sample form (adjust to match your actual form)
class ChatForm(forms.Form):
    user_input = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'placeholder': 'Type your message...',
            'rows': 3
        }),
        label='Message'
    )

# Sample model for storing chat history (adjust to match your actual model)
class ChatMessage:
    def __init__(self, sender, text, timestamp=None):
        self.sender = sender
        self.text = text
        self.timestamp = timestamp or timezone.now()

def chat_view(request):
    # Get or initialize chat history from session
    chat_history = request.session.get('chat_history', [])
    
    if request.method == 'POST':
        form = ChatForm(request.POST)
        if form.is_valid():
            user_input = form.cleaned_data['user_input']
            
            # Add user message to chat history
            chat_history.append({
                'sender': 'human',
                'text': user_input,
                'timestamp': timezone.now().isoformat()
            })
            
            # Process AI response (replace this with your actual AI logic)
            ai_response = process_ai_response(user_input)
            
            # Add AI response to chat history
            chat_history.append({
                'sender': 'ai',
                'text': ai_response,
                'timestamp': timezone.now().isoformat()
            })
            
            # Save updated chat history to session
            request.session['chat_history'] = chat_history
            request.session.modified = True
            
            # Add success message
            messages.success(request, 'Message sent successfully!')
            
            # Redirect to prevent form resubmission on refresh
            return HttpResponseRedirect(request.path_info)
    else:
        form = ChatForm()
    
    # Convert chat history to proper objects for template
    chat_messages = []
    for msg in chat_history:
        chat_msg = ChatMessage(
            sender=msg['sender'],
            text=msg['text'],
            timestamp=timezone.datetime.fromisoformat(msg['timestamp']) if isinstance(msg['timestamp'], str) else msg['timestamp']
        )
        chat_messages.append(chat_msg)
    
    context = {
        'form': form,
        'chat_history': chat_messages,
    }
    
    return render(request, 'chat.html', context)

def process_ai_response(user_input):
    """
    Replace this function with your actual AI processing logic.
    This is just a placeholder that returns a sample response.
    """
    
    # Sample responses based on user input
    if 'active leads' in user_input.lower():
        return """
**📊 Active Leads Analysis**

Here are your top active leads requiring attention:

1. **TechCorp Solutions** - Score: 4.8/5.0
   - ERV Active • ₹5.2Cr revenue potential
   - Meeting scheduled for tomorrow
   - **Action**: Prepare proposal presentation

2. **Global Manufacturing Ltd** - Score: 4.6/5.0
   - NCA Hot • ₹8.1Cr revenue potential  
   - Intent score: 92/100
   - **Action**: Schedule executive meeting

3. **Innovation Enterprises** - Score: 4.4/5.0
   - ERV Winback • ₹3.8Cr revenue potential
   - Last contact: 2 days ago
   - **Action**: Follow up on previous discussion

**Recommendation**: Focus on TechCorp Solutions first due to imminent meeting and high engagement.
        """
    
    elif 'score' in user_input.lower() and any(word in user_input.lower() for word in ['nca', 'erv', 'cr', 'turnover']):
        return """
**📊 Lead Scoring Result**

**Final Ranking Score:** 4.2/5.0  
**Priority Level:** ⚡ **HIGH PRIORITY**

**Score Breakdown:**
- Beat Plan Category: 1.5 × 0.95 = 1.43
- Revenue Potential: 0.8 × 0.25 = 0.20  
- Intent Score: 0.89 × 0.25 = 0.22
- Meeting Status: 0.5 × 0.15 = 0.08

**Recommendation:** Add to high-priority pipeline and contact within 24-48 hours.

**Next Steps:**
1. Schedule discovery call
2. Prepare customized presentation  
3. Identify decision makers
4. Set timeline for proposal
        """
    
    elif 'performance' in user_input.lower() or 'report' in user_input.lower():
        return """
**📈 Performance Dashboard**

**This Week's Metrics:**
- New Leads Processed: 47
- Average Lead Score: 3.2/5.0
- Conversion Rate: +12% vs last week
- Pipeline Value: ₹156.3Cr

**Top Performing Segments:**
1. **ERV Active**: 38% conversion rate
2. **NCA Hot**: 31% conversion rate  
3. **ERV Winback**: 24% conversion rate

**Key Insights:**
- High-value prospects (₹50Cr+) showing 15% increase in engagement
- Territory North showing strongest performance
- Meeting-to-close ratio improved by 18%

**Recommendations:**
1. Increase focus on ERV Active segment
2. Implement nurturing campaign for NCA Cold leads
3. Replicate North territory strategies in other regions
        """
    
    else:
        return f"""
**🤖 AI Assistant Response**

I understand you're asking about: "{user_input}"

I can help you with:
- **Lead Scoring**: Analyze and score new prospects
- **Performance Analysis**: Generate reports and insights  
- **Priority Assessment**: Identify high-value opportunities
- **Risk Evaluation**: Flag at-risk leads and suggest actions
- **Strategic Recommendations**: Provide data-driven insights

Try asking me something like:
- "Score this lead: NCA Hot, ₹15Cr turnover, 89 intent score"
- "Show me active leads with high priority"
- "Generate performance report for this week"
- "Which leads need immediate attention?"

How can I assist you today?
        """

# Alternative view using class-based view (if you prefer)
from django.views.generic import TemplateView

class ChatView(TemplateView):
    template_name = 'chat.html'
    form_class = ChatForm
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = self.form_class()
        
        # Get chat history from session
        chat_history = self.request.session.get('chat_history', [])
        chat_messages = []
        for msg in chat_history:
            chat_msg = ChatMessage(
                sender=msg['sender'],
                text=msg['text'],
                timestamp=timezone.datetime.fromisoformat(msg['timestamp']) if isinstance(msg['timestamp'], str) else msg['timestamp']
            )
            chat_messages.append(chat_msg)
        
        context['chat_history'] = chat_messages
        return context
    
    def post(self, request, *args, **kwargs):
        form = self.form_class(request.POST)
        if form.is_valid():
            user_input = form.cleaned_data['user_input']
            
            # Get or initialize chat history
            chat_history = request.session.get('chat_history', [])
            
            # Add user message
            chat_history.append({
                'sender': 'human',
                'text': user_input,
                'timestamp': timezone.now().isoformat()
            })
            
            # Process AI response
            ai_response = process_ai_response(user_input)
            
            # Add AI response
            chat_history.append({
                'sender': 'ai',
                'text': ai_response,
                'timestamp': timezone.now().isoformat()
            })
            
            # Save to session
            request.session['chat_history'] = chat_history
            request.session.modified = True
            
            messages.success(request, 'Message processed successfully!')
            
            # Redirect to prevent resubmission
            return HttpResponseRedirect(request.path_info)
        
        # If form is invalid, show the form with errors
        context = self.get_context_data()
        context['form'] = form
        return self.render_to_response(context)  
