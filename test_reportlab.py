from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def test_pdf():
    c = canvas.Canvas("test.pdf", pagesize=letter)
    c.drawString(100, 750, "Hello, World!")
    c.showPage()
    c.save()

if __name__ == "__main__":
    test_pdf()
