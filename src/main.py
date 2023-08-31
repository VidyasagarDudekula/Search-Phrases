from nltk import word_tokenize, pos_tag
from flair.data import Sentence
from flair.nn import Classifier
from utils import softClean, rankPhrases
import re
import flair
import torch
import time


class searchPhrases:
    def __init__(self):
        self.forbidden_pos = ['POS']
        self.load_model()

    def load_model(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        flair.device = device
        self.tagger = Classifier.load('chunk')

    def modify_text(self, text):
        text = text.replace('"', '')
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        ind = 0
        while ind < len(pos_tags) and pos_tags[ind][1] in self.forbidden_pos:
            tokens[ind] = str('')
            ind += 1
        if ind == 0:
            return text
        modified_text = ' '.join(tokens)
        text1 = re.sub(r"\s+", " ", modified_text).strip()
        return text1

    def valid_chunk(self, text):
        if len(text.split()) <= 2:
            return False
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        for token, pos in pos_tags:
            if pos not in ['NN', 'NNP', 'NNS', 'DT', 'CD', 'JJ']:
                return False
            break
        return True

    def removeLast(self, tokens):
        while tokens != [] and tokens[-1][-1] in ['PP', 'VP']:
            tokens = tokens[:-1]
        tokens = [a for a, _ in tokens]
        result = "".join(tokens).strip()
        result = result.replace('###', " ")
        return result.strip()

    def slidingWindowChunks(self, sentence, factor=8):
        dummy = []
        prev = False
        for chunk in sentence.get_labels():
            if chunk.value == 'NP':
                s = str(chunk)
                start = int(s[s.index('[')+1:s.index(':')])
                end = int(s[s.index(':')+1:s.index(']')])
                dummy.append((chunk.data_point.text, chunk.score, start, end, chunk.value))
                prev = True
            elif chunk.value in ['PP', 'VP'] and prev:
                t = dummy[-1]
                s = str(chunk)
                st = int(s[s.index('[')+1:s.index(':')])
                if st != t[3]:
                    prev = False
                    continue
                start = st
                end = int(s[s.index(':')+1:s.index(']')])
                dummy.append((chunk.data_point.text, chunk.score, start, end, chunk.value))
            else:
                prev = False
        result = []
        prev = ""
        end = dummy[0][2]
        for i in range(len(dummy)):
            if dummy[i][-1] in ['PP', 'VP']:
                continue
            j = i
            counter = 0
            value = []
            end = dummy[j][2]
            while j < len(dummy) and dummy[j][2] == end and len(dummy[j][0].split()) + counter <= factor:
                counter += len(dummy[j][0].split())
                space = "###"
                if dummy[j][0].strip()[0] == "'":
                    space = ""
                value.append([space + dummy[j][0].strip(), dummy[j][-1]])
                end = dummy[j][3]
                j += 1
            result.append(self.removeLast(value))
            i += 1
        return result

    def remove_overlap(self, result):
        dummy = []
        for i in range(len(result)):
            flag = True
            for j in range(len(result)):
                if i != j:
                    if result[i].lower() in result[j].lower():
                        flag = False
                        break
            if flag:
                dummy.append(result[i])
        return list(set(dummy))

    def get_noun_chunks(self, text, factor=8):
        text = softClean(text)
        result = []
        seen = []
        sentence = Sentence(text)
        self.tagger.predict(sentence)
        tokens = []
        tokens.extend(self.slidingWindowChunks(sentence, factor))
        tokens = list(set(tokens))
        for chunk in tokens:
            c_text = chunk
            c_text = self.modify_text(c_text)
            if self.valid_chunk(c_text) and c_text.lower().strip() not in seen:
                result.append(c_text)
                seen.append(c_text.lower().strip())
        result = self.remove_overlap(result)
        torch.cuda.empty_cache()
        result = rankPhrases(text, result)
        return result


if __name__ == '__main__':
    text = """
        Numerous conservative social media users have shared a video with claims that it shows President Joe Biden's son, Hunter Biden, "sniffing" an unknown substance at the White House—although that is far from evident from the images. The unverified claims come shortly after the White House was evacuated following the discovery of cocaine in the West Wing. Newsweek reached out to the White House via email for comment and to a legal firm that has represented Hunter Biden. Steve Guest, a former communications adviser to Texas Senator Ted Cruz, shared a video showing the Biden family at the White House on the Fourth of July, asking followers to provide their own captions. The tweet prompted several other Twitter users to share the video with many suggesting that it showed Hunter Biden using cocaine. Caption contest: pic.twitter.com/LNX9hGCH7V. Twitter user Team USA wrote, "Is Hunter Biden sniffing something here?". Is Hunter Biden sniffing something here?pic.twitter.com/W0URZVi1W8. Conservative Twitter user Rogan O'Handley shared the video and wrote, "Nothing to see here. Just a video of Hunter Biden allegedly doing a bump of cocaine at the White House in front of children. But don't worry - the media said the bag of blow found at the WH wasn't Hunter's!". Nothing to see hereJust a video of Hunter Biden allegedly doing a bump of cocaine at the White House in front of children But don’t worry - the media said the bag of blow found at the WH wasn’t Hunter’s! pic.twitter.com/Bt5hCT2ghf. Anthony Guglielmi, a spokesman for the Secret Service, told Newsweek on Monday that cocaine was found in the White House by officials with the Uniformed Division of the Secret Service as they were conducting routine checks throughout the building. A source involved in the investigation also previously told Newsweek that the cocaine was found in a "work area" of the White House's West Wing. While officials never said who left the cocaine in the White House's West Wing, a number of social media users joked that it could be connected to Hunter Biden, who has previously admitted having had an addiction to crack and cocaine. Several other social media users also suggested that the video could show Hunter Biden using cocaine. "WOW - Check out this video footage of Hunter Biden sniffing something near children during the 4th of July party at the White House. What other footage of Hunter Biden is the White House hiding from the American people?" Twitter user The Trump Train wrote. Conservative commentator Ian Miles Cheong shared the video and wrote, "What's Hunter doing? Is that what I think it is?". What's Hunter doing? Is that what I think it is? pic.twitter.com/bd1lOATIai. Twitter user Matt Wallace wrote, "NEW VIDEO APPEARS TO SHOW HUNTER BIDEN SNIFFING COCAINE IN FRONT OF KIDS AT THE WHITE HOUSE.". 🚨 NEW VIDEO APPEARS TO SHOW HUNTER BIDEN SNIFFING COCAINE IN FRONT OF KIDS AT THE WHITE HOUSE ⚠️ pic.twitter.com/Tp1fk5fqW5.
    """
    obj = searchPhrases()
    start_time = time.time()
    result = obj.get_noun_chunks(text)
    total_time = str(round(time.time()-start_time, 2))
    print("Time to generate the output:-", total_time+"s")
    for item in result:
        print(item)
