import streamlit as st
import random
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re


class PrescriptionVerificationSystem:
    def __init__(self):
        # Load the sentence transformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Define medical knowledge base for similarity matching
        self.drug_database = [
            "paracetamol acetaminophen pain relief fever reducer",
            "aspirin pain relief anti-inflammatory blood thinner",
            "ibuprofen anti-inflammatory pain relief fever reducer",
            "amoxicillin antibiotic bacterial infection treatment",
            "lisinopril ACE inhibitor hypertension blood pressure",
            "metformin diabetes blood sugar glucose control",
            "atorvastatin statin cholesterol lipid lowering",
            "omeprazole proton pump inhibitor acid reflux GERD",
            "warfarin blood thinner anticoagulant clotting prevention",
            "hydrochlorothiazide diuretic blood pressure fluid retention"
        ]

        self.contraindications = [
            "aspirin allergy salicylate hypersensitivity bleeding disorder",
            "penicillin allergy amoxicillin beta-lactam antibiotic reaction",
            "warfarin bleeding disorder hemorrhage anticoagulant risk",
            "metformin kidney disease renal impairment diabetes medication",
            "ACE inhibitor pregnancy angioedema kidney disease",
            "ibuprofen kidney disease heart failure bleeding ulcer"
        ]

        # Pre-compute embeddings for efficiency
        self.drug_embeddings = self.model.encode(self.drug_database)
        self.contraindication_embeddings = self.model.encode(self.contraindications)

    def extract_medications(self, prescription_text):
        """Extract medications using semantic similarity"""
        # Encode the prescription text
        prescription_embedding = self.model.encode([prescription_text])

        # Calculate similarities with drug database
        similarities = cosine_similarity(prescription_embedding, self.drug_embeddings)[0]

        # Find medications with similarity above threshold
        threshold = 0.3
        detected_drugs = []
        for i, similarity in enumerate(similarities):
            if similarity > threshold:
                drug_info = self.drug_database[i].split()[0]  # Get the main drug name
                detected_drugs.append({
                    'name': drug_info,
                    'confidence': float(similarity),
                    'description': self.drug_database[i]
                })

        return sorted(detected_drugs, key=lambda x: x['confidence'], reverse=True)

    def check_contraindications(self, prescription_text, patient_allergies, patient_conditions):
        """Check for contraindications using semantic analysis"""
        # Combine patient info for analysis
        patient_profile = f"allergies: {' '.join(patient_allergies)} conditions: {' '.join(patient_conditions)}"
        combined_text = f"{prescription_text} {patient_profile}"

        # Encode the combined text
        combined_embedding = self.model.encode([combined_text])

        # Calculate similarities with contraindications
        similarities = cosine_similarity(combined_embedding, self.contraindication_embeddings)[0]

        # Find potential contraindications
        threshold = 0.4
        warnings = []
        for i, similarity in enumerate(similarities):
            if similarity > threshold:
                warnings.append({
                    'warning': self.contraindications[i],
                    'severity': 'High' if similarity > 0.7 else 'Medium' if similarity > 0.5 else 'Low',
                    'confidence': float(similarity)
                })

        return sorted(warnings, key=lambda x: x['confidence'], reverse=True)

    def analyze_dosage_patterns(self, prescription_text):
        """Analyze dosage patterns using regex and semantic understanding"""
        dosage_patterns = [
            r'(\d+)\s*(mg|g|ml|mcg|units?)',
            r'(once|twice|thrice|\d+\s*times?)\s*(daily|per day|a day)',
            r'(morning|evening|night|bedtime|before meals|after meals)',
            r'(take|administer|give)\s*(\d+)'
        ]

        extracted_dosages = []
        for pattern in dosage_patterns:
            matches = re.findall(pattern, prescription_text.lower())
            extracted_dosages.extend(matches)

        # Use semantic similarity to validate dosage instructions
        standard_instructions = [
            "take one tablet twice daily with food",
            "administer 500mg every 8 hours",
            "take before bedtime as needed",
            "apply topically three times daily"
        ]

        instruction_embeddings = self.model.encode(standard_instructions)
        prescription_embedding = self.model.encode([prescription_text])
        similarities = cosine_similarity(prescription_embedding, instruction_embeddings)[0]

        best_match_idx = np.argmax(similarities)
        instruction_clarity = float(similarities[best_match_idx])

        return {
            'extracted_dosages': extracted_dosages,
            'instruction_clarity': instruction_clarity,
            'suggested_format': standard_instructions[best_match_idx] if instruction_clarity > 0.3 else None
        }

    def calculate_prescription_quality(self, prescription_text):
        """Calculate overall prescription quality using semantic analysis"""
        quality_indicators = [
            "clear medication name and dosage specified",
            "frequency of administration clearly stated",
            "duration of treatment mentioned",
            "special instructions for administration provided",
            "patient monitoring requirements specified"
        ]

        indicator_embeddings = self.model.encode(quality_indicators)
        prescription_embedding = self.model.encode([prescription_text])
        similarities = cosine_similarity(prescription_embedding, indicator_embeddings)[0]

        quality_score = np.mean(similarities)
        quality_breakdown = {
            indicator: float(sim) for indicator, sim in zip(quality_indicators, similarities)
        }

        return quality_score, quality_breakdown

    def verify_prescription(self, prescription_data):
        """Enhanced verification using sentence transformers"""
        prescription_text = prescription_data.get('text', '')
        patient_age = prescription_data.get('patient_info', {}).get('age', 0)
        patient_conditions = prescription_data.get('patient_info', {}).get('conditions', [])
        patient_allergies = prescription_data.get('patient_info', {}).get('allergies', [])

        # Extract medications using semantic similarity
        detected_medications = self.extract_medications(prescription_text)

        # Check contraindications
        contraindication_warnings = self.check_contraindications(
            prescription_text, patient_allergies, patient_conditions
        )

        # Analyze dosage patterns
        dosage_analysis = self.analyze_dosage_patterns(prescription_text)

        # Calculate prescription quality
        quality_score, quality_breakdown = self.calculate_prescription_quality(prescription_text)

        # Determine validity based on multiple factors
        is_valid = True
        risk_factors = []

        # Age-based checks
        if patient_age > 65:
            risk_factors.append("Elderly patient - consider dosage adjustment and drug interactions")
        elif patient_age < 18:
            risk_factors.append("Pediatric patient - verify age-appropriate dosing")

        # Contraindication checks
        if contraindication_warnings:
            high_severity_warnings = [w for w in contraindication_warnings if w['severity'] == 'High']
            if high_severity_warnings:
                is_valid = False
                risk_factors.append(f"High-risk contraindications detected: {len(high_severity_warnings)} warnings")
            else:
                risk_factors.extend([f"Potential contraindication: {w['warning'][:50]}..."
                                     for w in contraindication_warnings[:2]])

        # Quality-based checks
        if quality_score < 0.3:
            risk_factors.append("Prescription lacks clarity in dosage or administration instructions")

        if dosage_analysis['instruction_clarity'] < 0.2:
            risk_factors.append("Dosage instructions are unclear or non-standard")

        # Generate recommendations
        recommendations = []
        if patient_age > 65 and detected_medications:
            recommendations.append("Consider renal function assessment for elderly patient")

        if quality_score < 0.5:
            recommendations.append("Consider adding more detailed administration instructions")

        if dosage_analysis['suggested_format']:
            recommendations.append(f"Suggested format: {dosage_analysis['suggested_format']}")

        if not risk_factors:
            recommendations.append("Prescription appears appropriate for patient profile")

        if detected_medications:
            med_names = [med['name'] for med in detected_medications[:3]]
            recommendations.append(f"Monitor patient response to: {', '.join(med_names)}")

        # Calculate overall confidence
        confidence_factors = [
            quality_score,
            1.0 - (len(contraindication_warnings) * 0.2),
            dosage_analysis['instruction_clarity'],
            1.0 if detected_medications else 0.3
        ]
        confidence_score = np.mean([max(0, min(1, factor)) for factor in confidence_factors])

        return {
            "verification_status": "Valid" if is_valid else "Invalid",
            "is_valid": is_valid,
            "confidence_score": float(confidence_score),
            "risk_assessment": {
                "risk_level": "Low" if len(risk_factors) <= 1 else "Medium" if len(risk_factors) <= 3 else "High",
                "risk_factors": risk_factors if risk_factors else ["No significant risks identified"]
            },
            "medical_analysis": f"Semantic analysis of prescription for {patient_age}-year-old patient. " +
                                f"Detected {len(detected_medications)} medications with {len(contraindication_warnings)} potential warnings. " +
                                f"Prescription quality score: {quality_score:.2f}",
            "text_analysis": {
                "detected_medications": detected_medications,
                "contraindication_warnings": contraindication_warnings,
                "dosage_analysis": dosage_analysis,
                "quality_score": float(quality_score),
                "quality_breakdown": {k: float(v) for k, v in quality_breakdown.items()},
                "sentiment": {
                    "label": "Professional" if quality_score > 0.5 else "Needs Improvement",
                    "score": float(quality_score)
                }
            },
            "recommendations": recommendations,
            "timestamp": st.session_state.get('timestamp', 'N/A')
        }


# Initialize system with caching
@st.cache_resource
def load_verifier():
    return PrescriptionVerificationSystem()


# Streamlit UI
st.set_page_config(page_title="🏥 AI Prescription Verification", layout="wide")
st.title("🏥 AI-Powered Medical Prescription Verification System")
st.markdown("*Enhanced with Sentence Transformers for semantic analysis*")

# Add model info
with st.sidebar:
    st.header("🤖 Model Information")
    st.info("Using `sentence-transformers/all-MiniLM-L6-v2` for semantic text analysis")
    st.markdown("""
    **Features:**
    - Semantic medication detection
    - Contraindication analysis
    - Prescription quality assessment
    - Dosage pattern recognition
    """)

st.markdown("---")

# Load the system
with st.spinner("Loading AI model..."):
    verifier = load_verifier()

# Input area
st.subheader("📋 Enter Prescription Details")

with st.form("prescription_form"):
    col1, col2 = st.columns([2, 1])

    with col1:
        prescription_id = st.text_input("Prescription ID", value="RX_AI_001")
        prescription_text = st.text_area(
            "Prescription Text",
            height=150,
            value="Take Paracetamol 500mg twice daily after meals for 5 days. Monitor for any adverse reactions."
        )

    with col2:
        age = st.number_input("Patient Age", min_value=0, value=35)
        conditions_input = st.text_input("Medical Conditions", value="hypertension, diabetes")
        allergies_input = st.text_input("Known Allergies", value="penicillin")

    submit_btn = st.form_submit_button("🔍 Verify Prescription", type="primary")

if submit_btn:
    patient_info = {
        'age': age,
        'conditions': [c.strip() for c in conditions_input.split(',') if c.strip()],
        'allergies': [a.strip() for a in allergies_input.split(',') if a.strip()]
    }

    prescription_data = {
        'id': prescription_id,
        'text': prescription_text,
        'patient_info': patient_info
    }

    with st.spinner("🧠 Performing semantic analysis..."):
        result = verifier.verify_prescription(prescription_data)

    # Display results
    if isinstance(result, dict):
        # Status and confidence
        status = result.get('verification_status', 'Unknown')
        if status == 'Valid':
            st.success(f"✅ Verification Complete — Status: {status}")
        else:
            st.error(f"❌ Verification Complete — Status: {status}")

        col1, col2, col3 = st.columns(3)
        with col1:
            confidence = result.get('confidence_score', 0)
            st.metric("Confidence Score", f"{confidence:.1%}")

        with col2:
            quality_score = result.get('text_analysis', {}).get('quality_score', 0)
            st.metric("Prescription Quality", f"{quality_score:.1%}")

        with col3:
            risk_level = result.get("risk_assessment", {}).get("risk_level", "Unknown")
            st.metric("Risk Level", risk_level)

        # Detected medications
        detected_meds = result.get("text_analysis", {}).get("detected_medications", [])
        if detected_meds:
            with st.expander("💊 Detected Medications", expanded=True):
                for med in detected_meds[:5]:  # Show top 5
                    st.markdown(f"**{med['name'].title()}** (Confidence: {med['confidence']:.1%})")
                    st.caption(med['description'])

        # Risk Assessment
        with st.expander("🛡️ Risk Assessment"):
            st.markdown(f"**Risk Level:** {risk_level}")
            risk_factors = result.get("risk_assessment", {}).get("risk_factors", [])
            for factor in risk_factors:
                st.markdown(f"- {factor}")

        # Contraindication warnings
        warnings = result.get("text_analysis", {}).get("contraindication_warnings", [])
        if warnings:
            with st.expander("⚠️ Contraindication Warnings"):
                for warning in warnings:
                    severity_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
                    st.markdown(
                        f"{severity_color.get(warning['severity'], '⚪')} **{warning['severity']} Risk** (Confidence: {warning['confidence']:.1%})")
                    st.caption(warning['warning'])

        # Quality breakdown
        with st.expander("📊 Prescription Quality Analysis"):
            quality_breakdown = result.get("text_analysis", {}).get("quality_breakdown", {})
            for aspect, score in quality_breakdown.items():
                score = max(0.0, min(1.0, score))
                st.progress(score, text=f"{aspect.title()}: {score:.1%}")

        # Dosage analysis
        dosage_info = result.get("text_analysis", {}).get("dosage_analysis", {})
        if dosage_info:
            with st.expander("💉 Dosage Analysis"):
                st.markdown(f"**Instruction Clarity:** {dosage_info.get('instruction_clarity', 0):.1%}")
                if dosage_info.get('suggested_format'):
                    st.markdown(f"**Suggested Format:** {dosage_info['suggested_format']}")
                if dosage_info.get('extracted_dosages'):
                    st.markdown("**Extracted Dosages:**")
                    st.json(dosage_info['extracted_dosages'])

        # Recommendations
        with st.expander("💡 AI Recommendations"):
            recommendations = result.get("recommendations", [])
            if recommendations:
                for rec in recommendations:
                    st.markdown(f"- {rec}")
            else:
                st.write("No specific recommendations available")

        # Medical Analysis
        with st.expander("🧠 Detailed Analysis"):
            medical_analysis = result.get("medical_analysis", "No analysis available")
            st.write(medical_analysis)

    else:
        st.error(f"Error: Expected dictionary, but got {type(result)}")
        st.write("Raw result:", result)

st.markdown("---")
st.caption("🔬 Powered by Sentence Transformers AI | For educational/demo purposes only - Not for clinical use")