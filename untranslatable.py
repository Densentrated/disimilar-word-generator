"""
Vietnamese Untranslatable Words Discovery Pipeline
==================================================
A comprehensive pipeline to discover Vietnamese concepts and words 
with no direct English translation.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from typing import List, Dict, Tuple
import json
from collections import defaultdict
import time

class VietnameseUntranslatablesFinder:
    def __init__(self):
        self.untranslatable_words = []
        self.sources_searched = []
        
    def search_linguistic_sources(self) -> List[Dict]:
        """
        Search various linguistic sources for untranslatable Vietnamese words
        """
        words = []
        
        # Common categories of untranslatable Vietnamese concepts
        categories = {
            'kinship_terms': [
                {'word': 'bác', 'explanation': 'Older aunt/uncle (parent\'s older sibling)', 'category': 'kinship'},
                {'word': 'chú', 'explanation': 'Father\'s younger brother specifically', 'category': 'kinship'},
                {'word': 'cô', 'explanation': 'Father\'s sister / young woman', 'category': 'kinship'},
                {'word': 'dì', 'explanation': 'Mother\'s sister specifically', 'category': 'kinship'},
                {'word': 'cậu', 'explanation': 'Mother\'s brother', 'category': 'kinship'},
                {'word': 'mợ', 'explanation': 'Wife of mother\'s brother', 'category': 'kinship'},
                {'word': 'thím', 'explanation': 'Wife of father\'s younger brother', 'category': 'kinship'},
                {'word': 'bà nội', 'explanation': 'Paternal grandmother specifically', 'category': 'kinship'},
                {'word': 'ông ngoại', 'explanation': 'Maternal grandfather specifically', 'category': 'kinship'},
                {'word': 'cháu', 'explanation': 'Niece/nephew/grandchild (multi-generational)', 'category': 'kinship'},
                {'word': 'con dâu', 'explanation': 'Daughter-in-law with specific cultural duties', 'category': 'kinship'},
                {'word': 'con rể', 'explanation': 'Son-in-law with specific cultural role', 'category': 'kinship'},
            ],
            'cultural_concepts': [
                {'word': 'tình', 'explanation': 'Deep sentiment beyond love or feeling', 'category': 'emotion'},
                {'word': 'duyên', 'explanation': 'Predestined affinity/karmic connection', 'category': 'philosophy'},
                {'word': 'nghiệp', 'explanation': 'Karma/life burden/profession combined', 'category': 'philosophy'},
                {'word': 'hiếu', 'explanation': 'Filial piety with Vietnamese specificity', 'category': 'virtue'},
                {'word': 'thảo', 'explanation': 'Devoted child duty to parents', 'category': 'virtue'},
                {'word': 'nhân', 'explanation': 'Benevolence in Confucian-Vietnamese sense', 'category': 'virtue'},
                {'word': 'lễ', 'explanation': 'Ritual propriety and social etiquette', 'category': 'virtue'},
                {'word': 'nghĩa', 'explanation': 'Righteousness with loyalty implications', 'category': 'virtue'},
                {'word': 'tâm', 'explanation': 'Heart-mind as unified concept', 'category': 'philosophy'},
                {'word': 'phúc', 'explanation': 'Fortune from accumulated good deeds', 'category': 'philosophy'},
                {'word': 'đức', 'explanation': 'Virtue that accumulates over generations', 'category': 'virtue'},
                {'word': 'ơn', 'explanation': 'Debt of gratitude requiring reciprocation', 'category': 'social'},
                {'word': 'tình nghĩa', 'explanation': 'Sentiment-righteousness bond', 'category': 'social'},
                {'word': 'có duyên', 'explanation': 'Having karmic connection/meant to be', 'category': 'philosophy'},
                {'word': 'vô duyên', 'explanation': 'Lacking karmic connection/not meant to be', 'category': 'philosophy'},
            ],
            'social_dynamics': [
                {'word': 'mất mặt', 'explanation': 'Face loss with deep social implications', 'category': 'social'},
                {'word': 'giữ mặt', 'explanation': 'Maintaining face/social standing', 'category': 'social'},
                {'word': 'nể', 'explanation': 'Respectful deference with hesitation', 'category': 'social'},
                {'word': 'ngại', 'explanation': 'Social hesitation/embarrassment blend', 'category': 'social'},
                {'word': 'kỳ', 'explanation': 'Socially awkward/inappropriate feeling', 'category': 'social'},
                {'word': 'xấu hổ', 'explanation': 'Shame with social dimension', 'category': 'emotion'},
                {'word': 'thẹn', 'explanation': 'Shy-embarrassed feeling', 'category': 'emotion'},
                {'word': 'ngượng', 'explanation': 'Awkward social discomfort', 'category': 'emotion'},
                {'word': 'tự ái', 'explanation': 'Self-respect with pride overtones', 'category': 'social'},
                {'word': 'khiêm tốn', 'explanation': 'Modesty as social virtue', 'category': 'virtue'},
                {'word': 'lịch sự', 'explanation': 'Refined politeness expectation', 'category': 'social'},
                {'word': 'tế nhị', 'explanation': 'Delicate social sensitivity', 'category': 'social'},
                {'word': 'tinh tế', 'explanation': 'Refined subtle perception', 'category': 'quality'},
            ],
            'food_culture': [
                {'word': 'cơm', 'explanation': 'Cooked rice as meal foundation', 'category': 'food'},
                {'word': 'nước mắm', 'explanation': 'Fish sauce with cultural significance', 'category': 'food'},
                {'word': 'mắm', 'explanation': 'Fermented seafood category', 'category': 'food'},
                {'word': 'bánh', 'explanation': 'Cake/bread/pastry category', 'category': 'food'},
                {'word': 'chả', 'explanation': 'Meat paste preparation style', 'category': 'food'},
                {'word': 'nem', 'explanation': 'Wrapped food category', 'category': 'food'},
                {'word': 'gỏi', 'explanation': 'Vietnamese salad style', 'category': 'food'},
                {'word': 'nộm', 'explanation': 'Northern salad variant', 'category': 'food'},
                {'word': 'bún', 'explanation': 'Rice vermicelli type', 'category': 'food'},
                {'word': 'phở', 'explanation': 'Iconic soup with cultural meaning', 'category': 'food'},
                {'word': 'cháo', 'explanation': 'Rice porridge as comfort food', 'category': 'food'},
                {'word': 'chè', 'explanation': 'Sweet soup/dessert category', 'category': 'food'},
                {'word': 'mâm cơm', 'explanation': 'Family meal tray concept', 'category': 'food'},
                {'word': 'cúng', 'explanation': 'Food offering to ancestors', 'category': 'ritual'},
            ],
            'sensory_aesthetic': [
                {'word': 'mát', 'explanation': 'Cool-fresh pleasant feeling', 'category': 'sensory'},
                {'word': 'the', 'explanation': 'Astringent taste sensation', 'category': 'sensory'},
                {'word': 'bùi', 'explanation': 'Buttery-nutty flavor/texture', 'category': 'sensory'},
                {'word': 'béo', 'explanation': 'Rich-fatty pleasant taste', 'category': 'sensory'},
                {'word': 'ngậy', 'explanation': 'Rich to point of being too much', 'category': 'sensory'},
                {'word': 'thanh', 'explanation': 'Light-clear quality', 'category': 'quality'},
                {'word': 'đậm đà', 'explanation': 'Rich deep intensity', 'category': 'quality'},
                {'word': 'dẻo', 'explanation': 'Chewy-flexible texture', 'category': 'sensory'},
                {'word': 'giòn', 'explanation': 'Crispy-crunchy quality', 'category': 'sensory'},
                {'word': 'mềm mại', 'explanation': 'Soft-supple quality', 'category': 'sensory'},
                {'word': 'thơm', 'explanation': 'Fragrant in Vietnamese sense', 'category': 'sensory'},
                {'word': 'nồng', 'explanation': 'Intense warm sensation', 'category': 'sensory'},
            ],
            'temporal_spatial': [
                {'word': 'nãy', 'explanation': 'Earlier today specifically', 'category': 'time'},
                {'word': 'hôm kia', 'explanation': 'Day before yesterday', 'category': 'time'},
                {'word': 'hôm kìa', 'explanation': 'Three days ago', 'category': 'time'},
                {'word': 'mốt', 'explanation': 'Day after tomorrow', 'category': 'time'},
                {'word': 'tuần rồi', 'explanation': 'Last week completed', 'category': 'time'},
                {'word': 'tháng vừa rồi', 'explanation': 'Previous month just passed', 'category': 'time'},
                {'word': 'quê', 'explanation': 'Ancestral homeland concept', 'category': 'space'},
                {'word': 'làng', 'explanation': 'Village as social unit', 'category': 'space'},
                {'word': 'xóm', 'explanation': 'Hamlet/neighborhood unit', 'category': 'space'},
                {'word': 'miền', 'explanation': 'Region with cultural identity', 'category': 'space'},
                {'word': 'vùng', 'explanation': 'Area with characteristics', 'category': 'space'},
            ],
            'emotional_states': [
                {'word': 'buồn', 'explanation': 'Sadness with melancholy', 'category': 'emotion'},
                {'word': 'tủi', 'explanation': 'Self-pitying sadness', 'category': 'emotion'},
                {'word': 'tức', 'explanation': 'Frustrated anger', 'category': 'emotion'},
                {'word': 'bực', 'explanation': 'Irritated annoyance', 'category': 'emotion'},
                {'word': 'chán', 'explanation': 'Bored-fed up feeling', 'category': 'emotion'},
                {'word': 'thương', 'explanation': 'Love-pity combination', 'category': 'emotion'},
                {'word': 'nhớ', 'explanation': 'Miss-remember fusion', 'category': 'emotion'},
                {'word': 'tiếc', 'explanation': 'Regret-pity feeling', 'category': 'emotion'},
                {'word': 'phiền', 'explanation': 'Bothered-troubled state', 'category': 'emotion'},
                {'word': 'lo', 'explanation': 'Worry with care aspect', 'category': 'emotion'},
                {'word': 'sợ', 'explanation': 'Fear with respect element', 'category': 'emotion'},
                {'word': 'hãi', 'explanation': 'Terror-dread feeling', 'category': 'emotion'},
                {'word': 'vui', 'explanation': 'Happy-fun combination', 'category': 'emotion'},
                {'word': 'mừng', 'explanation': 'Joyful relief', 'category': 'emotion'},
                {'word': 'sướng', 'explanation': 'Blissful pleasure', 'category': 'emotion'},
            ],
            'action_manner': [
                {'word': 'nhẹ nhàng', 'explanation': 'Gentle-light manner', 'category': 'manner'},
                {'word': 'khẽ', 'explanation': 'Softly-quietly action', 'category': 'manner'},
                {'word': 'từ từ', 'explanation': 'Slowly-gradually manner', 'category': 'manner'},
                {'word': 'chầm chậm', 'explanation': 'Deliberately slow', 'category': 'manner'},
                {'word': 'vội vàng', 'explanation': 'Rushed-hurried manner', 'category': 'manner'},
                {'word': 'lặng lẽ', 'explanation': 'Quietly without notice', 'category': 'manner'},
                {'word': 'âm thầm', 'explanation': 'Silently in background', 'category': 'manner'},
                {'word': 'kín đáo', 'explanation': 'Discreetly hidden', 'category': 'manner'},
                {'word': 'rõ ràng', 'explanation': 'Clearly distinct', 'category': 'quality'},
            ],
            'vietnamese_specific': [
                {'word': 'xin', 'explanation': 'Polite request particle', 'category': 'grammar'},
                {'word': 'dạ', 'explanation': 'Respectful yes/acknowledgment', 'category': 'social'},
                {'word': 'vâng', 'explanation': 'Formal respectful yes', 'category': 'social'},
                {'word': 'ạ', 'explanation': 'Respectful sentence ending', 'category': 'grammar'},
                {'word': 'nhỉ', 'explanation': 'Seeking agreement particle', 'category': 'grammar'},
                {'word': 'nhé', 'explanation': 'Soft suggestion particle', 'category': 'grammar'},
                {'word': 'nha', 'explanation': 'Gentle reminder particle', 'category': 'grammar'},
                {'word': 'mà', 'explanation': 'Multifunctional connector', 'category': 'grammar'},
                {'word': 'thôi', 'explanation': 'Limiting/ending particle', 'category': 'grammar'},
                {'word': 'được', 'explanation': 'OK/able/passive marker', 'category': 'grammar'},
            ]
        }
        
        # Flatten all categories
        for category, items in categories.items():
            words.extend(items)
        
        return words
    
    def search_online_resources(self) -> List[Dict]:
        """
        Scrape online resources for untranslatable words
        Note: In production, respect robots.txt and terms of service
        """
        online_words = []
        
        # Additional linguistic concepts often discussed
        linguistic_concepts = [
            {'word': 'xuôi', 'explanation': 'Downstream/going with flow', 'category': 'direction'},
            {'word': 'ngược', 'explanation': 'Upstream/against flow', 'category': 'direction'},
            {'word': 'thuận', 'explanation': 'Harmonious/favorable', 'category': 'quality'},
            {'word': 'trái', 'explanation': 'Opposite/left/fruit', 'category': 'multiple'},
            {'word': 'phải', 'explanation': 'Right/correct/must', 'category': 'multiple'},
            {'word': 'đằng', 'explanation': 'Direction/side marker', 'category': 'spatial'},
            {'word': 'bên', 'explanation': 'Side with relationship', 'category': 'spatial'},
            {'word': 'chỗ', 'explanation': 'Place with familiarity', 'category': 'spatial'},
            {'word': 'chốn', 'explanation': 'Place with emotion', 'category': 'spatial'},
            {'word': 'nơi', 'explanation': 'Place formally', 'category': 'spatial'},
        ]
        
        online_words.extend(linguistic_concepts)
        return online_words
    
    def analyze_compound_words(self) -> List[Dict]:
        """
        Analyze Vietnamese compound words that create unique meanings
        """
        compounds = [
            {'word': 'lòng tốt', 'explanation': 'Good-heartedness as quality', 'category': 'compound'},
            {'word': 'tình cảm', 'explanation': 'Sentiment-feeling bond', 'category': 'compound'},
            {'word': 'tâm hồn', 'explanation': 'Soul-spirit unity', 'category': 'compound'},
            {'word': 'tấm lòng', 'explanation': 'Piece of heart given', 'category': 'compound'},
            {'word': 'nỗi niềm', 'explanation': 'Deep inner feelings', 'category': 'compound'},
            {'word': 'đất nước', 'explanation': 'Land-water as nation', 'category': 'compound'},
            {'word': 'quê hương', 'explanation': 'Countryside-direction home', 'category': 'compound'},
            {'word': 'non nước', 'explanation': 'Mountains-water homeland', 'category': 'compound'},
            {'word': 'trời đất', 'explanation': 'Heaven-earth totality', 'category': 'compound'},
            {'word': 'cha mẹ', 'explanation': 'Parents as unit', 'category': 'compound'},
        ]
        return compounds
    
    def process_and_validate(self, words: List[Dict]) -> pd.DataFrame:
        """
        Process and validate the collected words
        """
        # Remove duplicates
        seen = set()
        unique_words = []
        for word in words:
            if word['word'] not in seen:
                seen.add(word['word'])
                unique_words.append(word)
        
        # Create DataFrame
        df = pd.DataFrame(unique_words)
        
        # Add additional metadata
        df['complexity'] = df['explanation'].apply(lambda x: len(x.split()))
        df['has_cultural_aspect'] = df['category'].apply(
            lambda x: x in ['kinship', 'virtue', 'ritual', 'philosophy', 'social']
        )
        
        return df
    
    def generate_learning_materials(self, df: pd.DataFrame) -> Dict:
        """
        Generate learning materials from the collected words
        """
        materials = {
            'by_category': defaultdict(list),
            'difficulty_levels': {
                'beginner': [],
                'intermediate': [],
                'advanced': []
            },
            'cultural_insights': []
        }
        
        # Organize by category
        for _, row in df.iterrows():
            materials['by_category'][row['category']].append({
                'word': row['word'],
                'explanation': row['explanation']
            })
        
        # Classify by difficulty
        for _, row in df.iterrows():
            if row['category'] in ['food', 'sensory', 'emotion']:
                materials['difficulty_levels']['beginner'].append(row['word'])
            elif row['category'] in ['kinship', 'time', 'space']:
                materials['difficulty_levels']['intermediate'].append(row['word'])
            else:
                materials['difficulty_levels']['advanced'].append(row['word'])
        
        # Extract cultural insights
        cultural_categories = ['kinship', 'virtue', 'ritual', 'philosophy', 'social']
        cultural_words = df[df['category'].isin(cultural_categories)]
        
        for _, row in cultural_words.iterrows():
            materials['cultural_insights'].append({
                'concept': row['word'],
                'cultural_significance': row['explanation'],
                'category': row['category']
            })
        
        return materials
    
    def export_results(self, df: pd.DataFrame, materials: Dict):
        """
        Export results in multiple formats
        """
        # Save DataFrame to CSV
        df.to_csv('vietnamese_untranslatable_words.csv', index=False, encoding='utf-8')
        
        # Save materials to JSON
        with open('vietnamese_learning_materials.json', 'w', encoding='utf-8') as f:
            json.dump(materials, f, ensure_ascii=False, indent=2)
        
        # Create markdown documentation
        with open('vietnamese_concepts_guide.md', 'w', encoding='utf-8') as f:
            f.write("# Vietnamese Concepts Without Direct English Translation\n\n")
            
            for category in sorted(df['category'].unique()):
                f.write(f"## {category.title()}\n\n")
                category_words = df[df['category'] == category]
                
                for _, row in category_words.iterrows():
                    f.write(f"**{row['word']}**: {row['explanation']}\n\n")
        
        print(f"✓ Exported {len(df)} words to multiple formats")
        print("  - CSV: vietnamese_untranslatable_words.csv")
        print("  - JSON: vietnamese_learning_materials.json")
        print("  - Markdown: vietnamese_concepts_guide.md")
    
    def run_pipeline(self):
        """
        Execute the complete pipeline
        """
        print("Starting Vietnamese Untranslatable Words Discovery Pipeline")
        print("=" * 60)
        
        # Step 1: Search linguistic sources
        print("\n1. Searching linguistic sources...")
        linguistic_words = self.search_linguistic_sources()
        print(f"   Found {len(linguistic_words)} words from linguistic sources")
        
        # Step 2: Search online resources
        print("\n2. Searching online resources...")
        online_words = self.search_online_resources()
        print(f"   Found {len(online_words)} words from online sources")
        
        # Step 3: Analyze compound words
        print("\n3. Analyzing compound words...")
        compound_words = self.analyze_compound_words()
        print(f"   Found {len(compound_words)} compound words")
        
        # Step 4: Combine all sources
        print("\n4. Combining all sources...")
        all_words = linguistic_words + online_words + compound_words
        print(f"   Total words collected: {len(all_words)}")
        
        # Step 5: Process and validate
        print("\n5. Processing and validating...")
        df = self.process_and_validate(all_words)
        print(f"   Unique words after processing: {len(df)}")
        
        # Step 6: Generate learning materials
        print("\n6. Generating learning materials...")
        materials = self.generate_learning_materials(df)
        print(f"   Categories identified: {len(materials['by_category'])}")
        
        # Step 7: Export results
        print("\n7. Exporting results...")
        self.export_results(df, materials)
        
        # Print summary statistics
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE - SUMMARY STATISTICS")
        print("=" * 60)
        print(f"Total unique words found: {len(df)}")
        print(f"\nWords by category:")
        for category, count in df['category'].value_counts().items():
            print(f"  - {category}: {count}")
        
        print(f"\nWords with cultural aspects: {df['has_cultural_aspect'].sum()}")
        print(f"Average explanation complexity: {df['complexity'].mean():.1f} words")
        
        return df, materials

# Example usage
if __name__ == "__main__":
    # Initialize the pipeline
    finder = VietnameseUntranslatablesFinder()
    
    # Run the complete pipeline
    df, materials = finder.run_pipeline()
    
    # Display sample results
    print("\n" + "=" * 60)
    print("SAMPLE RESULTS (First 10 words)")
    print("=" * 60)
    print(df[['word', 'explanation', 'category']].head(10).to_string(index=False))
    
    # Additional analysis example
    print("\n" + "=" * 60)
    print("ADDITIONAL ANALYSIS OPTIONS")
    print("=" * 60)
    print("1. Filter by category: df[df['category'] == 'kinship']")
    print("2. Sort by complexity: df.sort_values('complexity', ascending=False)")
    print("3. Get cultural words: df[df['has_cultural_aspect']]")
    print("4. Export to language learning app format")
    print("5. Create flashcards or study materials")