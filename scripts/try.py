#!/usr/bin/env python3

"""
Create a 12-week hangboard training plan spreadsheet with progress tracking columns.
"""

import pandas as pd

def main():
    # Data for each row in our training plan
    # Each list entry corresponds to a row in the final spreadsheet
    # You can adjust the text, phases, sets/reps, or other details as desired.

    weeks = [
        "Week 1", "Week 2", "Week 3", "Week 4",
        "Week 5", "Week 6", "Week 7", "Week 8",
        "Week 9", "Week 10", "Week 11", "Week 12"
    ]

    phases = [
        "Phase 1 (Familiarization)", "Phase 1 (Familiarization)", "Phase 1 (Familiarization)", "Phase 1 (Familiarization)",
        "Phase 2 (Increasing Intensity)", "Phase 2 (Increasing Intensity)", "Phase 2 (Increasing Intensity)", "Phase 2 (Increasing Intensity)",
        "Phase 3 (Peak)", "Phase 3 (Peak)", "Phase 4 (Taper)", "Phase 4 (Taper)"
    ]

    sessionA = [
        # Weeks 1-4: Base & Familiarization
        "Max Hangs (4–5 total)\n7–10s each @ 80–85% max\n2–3 min rest",
        "Max Hangs (4–5 total)\n7–10s each @ ~85% max\n2–3 min rest",
        "Max Hangs (5 total)\n7–10s each, +1–2 kg or slightly smaller edge",
        "Max Hangs (5 total)\n7–10s each, push intensity slightly",
        # Weeks 5-8: Increasing Intensity
        "Short Max Hangs (5 total)\n5–10s each @ 85–90% max\n+ 1–2 Repeaters sets",
        "Short Max Hangs (5–6 total)\n5–10s each\nTry smaller holds or + weight",
        "Max Hangs (6 total)\n5–10s each\nApproach ~90% max load",
        "Max Hangs (5–6 total)\n5–10s each\nMaybe smaller edge or + more weight",
        # Weeks 9-10: Peak
        "Maximal Hangs (4–5 total)\n5–7s each @ ~90–100% effort\nFull rest",
        "Maximal Hangs (4–5 total)\n5–7s each\nOne-arm if advanced or + weight",
        # Weeks 11-12: Taper
        "Reduced-Volume Max Hangs (3–4 total)\n5–7s each\nFocus on feeling fresh",
        "Reduced-Volume Max Hangs (2–3 total)\n5–7s each\nStay fresh"
    ]

    sessionB = [
        # Weeks 1-4
        "Repeaters (3 sets of 6× 7s:3s)\nComfortable edge\n2–3 min rest between sets",
        "Repeaters (3 sets of 6× 7s:3s)\nMaintain form\n2–3 min rest",
        "Repeaters (3–4 sets of 6× 7s:3s)\nFocus on consistent form",
        "Repeaters (3–4 sets) or 20s Hangs\nAnother grip type if time allows",
        # Weeks 5-8
        "Mixed: 1–2 sets of Repeaters or 2–3 long hangs (20s) on bigger edge",
        "Long/Tendon Hangs (2–3 total)\n30–45s each on easier edge",
        "Repeaters or Endurance Hangs (2–3 sets)\n6× 7s:3s @ ~70–80% max",
        "Long Hangs (2–3 total)\n20–30s each\n3–5 min rest",
        # Weeks 9-10
        "Short Endurance (1–2 sets)\n6× 7s:3s or skip if fatigued",
        "Short Endurance (1 set)\n6× 7s:3s or skip if fatigued",
        # Weeks 11-12
        "Optional 1 set of easy Repeaters or none\nFocus on performance climbing",
        "Optional short easy hangs or none\nPriority = project/comp performance"
    ]

    key_points = [
        # Additional notes about each week
        "Start easy, focus on technique & safe loading.\nAdd 1–2 easy bouldering sessions.",
        "Continue at moderate intensity, keep 1–2 min rest.\nLow-intensity climbs: V2–V3.",
        "Slightly increase difficulty or smaller holds.\nTrack finger comfort & recovery.",
        "Assess baseline progress. If any tweaks, rest.\nLight boulders or technique work.",
        "Increase hang intensity or volume. 2–3 sessions/week if comfortable.\nProject up to V5–V6.",
        "Build on short max hangs. Might add small weight.\nSteeper climbing sessions for strength.",
        "Higher intensity max hangs. Try pinches.\nLimit bouldering with adequate rest.",
        "Peak of intensity. Keep good rest intervals.\nFocus on quality over quantity.",
        "True high-intensity hangs, less total volume.\nTry V6–V7 limit boulders.",
        "Maintain near-max intensity with reduced volume.\nAvoid injuries, watch finger health.",
        "Reduce volume (~40%). 1–2 max hangs sets only.\nPerformance focus, fresh skin.",
        "Minimal fingerboard, just maintenance.\nPerform on your hardest projects or comps."
    ]

    # Create the base DataFrame
    df = pd.DataFrame({
        "Week": weeks,
        "Phase & Focus": phases,
        "Session A": sessionA,
        "Session B": sessionB,
        "Key Points / Additional Notes": key_points
    })

    # Add progress tracking columns
    # You can customize these as you prefer (columns for each session)
    df["Edge Size (mm) - A"] = ""
    df["Added Weight (kg) - A"] = ""
    df["RPE - A (1-10)"] = ""
    df["Notes - A"] = ""
    df["Edge Size (mm) - B"] = ""
    df["Added Weight (kg) - B"] = ""
    df["RPE - B (1-10)"] = ""
    df["Notes - B"] = ""

    # Save to Excel
    output_filename = "Hangboard_Training_Plan.xlsx"
    df.to_excel(output_filename, index=False)

    print(f"Success! Your hangboard training plan has been saved to '{output_filename}'.")
    print("Open it with Excel (or similar) to view and fill in your progress after each session.")

if __name__ == "__main__":
    main()
