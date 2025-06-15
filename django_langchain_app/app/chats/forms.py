from django import forms

class ChatForm(forms.Form):
    user_input = forms.CharField(
    label = "Your message",
    max_length = 500,
    widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Type your message...'
        })
    )