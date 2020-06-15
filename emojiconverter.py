def emojiconverter(message):
    output = ""
    words = message.split(' ')
    emojis = {":)":"😀", 
            ":(" : "😔 "}
    for word in words:
        output+=emojis.get(word,word)+" "
    return output

message = input(">")
output = emojiconverter(message)
print(output)