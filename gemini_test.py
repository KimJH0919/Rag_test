from langchain_google_genai import ChatGoogleGenerativeAI
import os


os.environ["GOOGLE_API_KEY"] = "AIzaSyBGeIEC8AitFlcxTnKO54P9dSRD09aJnpk"
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")
result = llm.invoke("동서울대학교 컴퓨터소프트웨어과의 위치는?")
print(result.content)
