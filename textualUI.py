from textual.app import App
from textual.widgets import Welcome, Button, Label


class TextualUI(App):

    def compose(self):
        yield Label("Do you want to continue?")
        yield Button("Yes","success",id="yes")
        yield Button("No","warning",id="no")

    def on_button_pressed(self,event:Button.Pressed):
        self.exit(event.button.id)


if __name__ == "__main__":
    result = TextualUI().run()
    print(result)