from benchita.template.template import Template, register_template

@register_template("default")
class DefaultTemplate(Template):
    def apply_chat_template(self, messages, **kwargs):
        ret = ""
        for message in messages:
            if message["role"] == "user":
                ret += f"USER: {message['content']}\n"
            elif message["role"] == "assistant":
                ret += f"ASSISTANT: {message['content']}\n"
            else:
                raise ValueError(f"Unknown role {message['role']}")
        
        ret += f"USER:"

        return ret 