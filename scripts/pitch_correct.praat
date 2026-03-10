# Reference-guided pitch correction using Praat PSOLA.
#
# Extracts F0 from both reference and user vocals, gently pulls
# the user's pitch toward the reference melody, then resynthesizes
# with overlap-add (PSOLA).
#
# Usage:
#   praat --run pitch_correct.praat user.wav ref.wav output.wav \
#         [strength] [tolerance_cents] [f0_min] [f0_max]

form Pitch correction
    sentence user_wav
    sentence ref_wav
    sentence output_wav
    real strength 0.6
    real tolerance_cents 50.0
    real f0_min 80.0
    real f0_max 600.0
endform

time_step = 0.01
tolerance_st = tolerance_cents / 100.0
smooth_radius = 3

# --- Load and prepare audio ---

snd_user = Read from file: user_wav$
selectObject: snd_user
nch_user = Get number of channels
if nch_user > 1
    snd_user_mono = Convert to mono
    removeObject: snd_user
    snd_user = snd_user_mono
endif

snd_ref = Read from file: ref_wav$
selectObject: snd_ref
nch_ref = Get number of channels
if nch_ref > 1
    snd_ref_mono = Convert to mono
    removeObject: snd_ref
    snd_ref = snd_ref_mono
endif

# Resample ref to match user if needed
selectObject: snd_user
sr = Get sampling frequency
selectObject: snd_ref
sr_ref = Get sampling frequency
if sr_ref <> sr
    snd_ref_rs = Resample: sr, 50
    removeObject: snd_ref
    snd_ref = snd_ref_rs
endif

# Workaround: save/reload to normalize internal format (matches Python version)
selectObject: snd_user
Scale peak: 0.99
tmp_user$ = temporaryDirectory$ + "/pc_user.wav"
Save as WAV file: tmp_user$
removeObject: snd_user
snd_user = Read from file: tmp_user$
deleteFile: tmp_user$

selectObject: snd_ref
Scale peak: 0.99
tmp_ref$ = temporaryDirectory$ + "/pc_ref.wav"
Save as WAV file: tmp_ref$
removeObject: snd_ref
snd_ref = Read from file: tmp_ref$
deleteFile: tmp_ref$

selectObject: snd_user
dur_user = Get total duration
writeInfoLine: "User: ", fixed$(dur_user, 1), "s @ ", fixed$(sr, 0), " Hz"

selectObject: snd_ref
dur_ref = Get total duration
appendInfoLine: "Ref:  ", fixed$(dur_ref, 1), "s @ ", fixed$(sr, 0), " Hz"

# --- Workaround: praat_barren crashes on To Manipulation when start time is exactly 0 ---
selectObject: snd_user
start_time = Get start time
if start_time = 0
    snd_user_trim = Extract part: 0.001, dur_user, "rectangular", 1, "no"
    removeObject: snd_user
    snd_user = snd_user_trim
    dur_user = dur_user - 0.001
endif

selectObject: snd_ref
start_time_ref = Get start time
if start_time_ref = 0
    snd_ref_trim = Extract part: 0.001, dur_ref, "rectangular", 1, "no"
    removeObject: snd_ref
    snd_ref = snd_ref_trim
    dur_ref = dur_ref - 0.001
endif

# --- Extract pitch contours ---

selectObject: snd_user
pitch_user = To Pitch: time_step, f0_min, f0_max

selectObject: snd_ref
pitch_ref = To Pitch: time_step, f0_min, f0_max

# --- Create Manipulation for PSOLA ---

selectObject: snd_user
manipulation = To Manipulation: time_step, f0_min, f0_max

selectObject: manipulation
pitch_tier = Extract pitch tier

# Clear existing pitch points
selectObject: pitch_tier
n_points = Get number of points
for i from n_points to 1
    Remove point: i
endfor

# --- Pass 1: compute corrected F0 per frame ---

duration = min(dur_user, dur_ref)
n_frames = floor(duration / time_step)

# Arrays for results
f0_out# = zero#(n_frames)
kind# = zero#(n_frames)
times# = zero#(n_frames)

n_voiced = 0
n_corrected = 0
n_skipped_close = 0
n_skipped_unvoiced = 0

for i from 1 to n_frames
    t = (i - 0.5) * time_step
    times#[i] = t

    selectObject: pitch_user
    f0_u = Get value at time: t, "Hertz", "Linear"

    selectObject: pitch_ref
    f0_r = Get value at time: t, "Hertz", "Linear"

    if f0_u = undefined or f0_r = undefined or f0_u = 0 or f0_r = 0
        n_skipped_unvoiced += 1
        if f0_u <> undefined and f0_u > 0
            f0_out#[i] = f0_u
            kind#[i] = 3
        else
            f0_out#[i] = 0
            kind#[i] = 0
        endif
    else
        n_voiced += 1
        st_u = 12 * log2(f0_u / 440)
        st_r = 12 * log2(f0_r / 440)

        diff = st_r - st_u
        if abs(diff) > 6
            st_r = st_r - 12 * round(diff / 12)
        endif

        diff = st_r - st_u

        if abs(diff) < tolerance_st
            f0_out#[i] = f0_u
            kind#[i] = 2
            n_skipped_close += 1
        else
            st_corrected = st_u + strength * diff
            f0_corrected = 440 * 2 ^ (st_corrected / 12)
            if f0_corrected < f0_min
                f0_corrected = f0_min
            endif
            if f0_corrected > f0_max
                f0_corrected = f0_max
            endif
            f0_out#[i] = f0_corrected
            kind#[i] = 1
            n_corrected += 1
        endif
    endif
endfor

# --- Pass 2: median smooth voiced F0 ---

smoothed# = zero#(n_frames)

for i from 1 to n_frames
    if f0_out#[i] <= 0
        smoothed#[i] = 0
    else
        lo = max(1, i - smooth_radius)
        hi = min(n_frames, i + smooth_radius)
        # Collect voiced values
        n_win = 0
        for j from lo to hi
            if f0_out#[j] > 0
                n_win += 1
                win_'n_win' = f0_out#[j]
            endif
        endfor
        if n_win = 0
            smoothed#[i] = f0_out#[i]
        else
            # Bubble sort for median (window is at most 7 elements)
            for a from 1 to n_win - 1
                for b from 1 to n_win - a
                    b1 = b + 1
                    if win_'b' > win_'b1'
                        tmp = win_'b'
                        win_'b' = win_'b1'
                        win_'b1' = tmp
                    endif
                endfor
            endfor
            mid = floor(n_win / 2) + 1
            smoothed#[i] = win_'mid'
        endif
    endif
endfor

# --- Pass 3: add smoothed points to pitch tier ---

selectObject: pitch_tier
for i from 1 to n_frames
    if kind#[i] > 0
        Add point: times#[i], smoothed#[i]
    endif
endfor

# --- Resynthesize ---

selectObject: manipulation
plusObject: pitch_tier
Replace pitch tier

selectObject: manipulation
result = Get resynthesis (overlap-add)

selectObject: result
Save as WAV file: output_wav$

# --- Stats ---

appendInfoLine: ""
appendInfoLine: "Frames:    ", n_frames
appendInfoLine: "Voiced:    ", n_voiced
if n_voiced > 0
    pct = n_corrected / n_voiced * 100
else
    pct = 0
endif
appendInfoLine: "Corrected: ", n_corrected, " (", fixed$(pct, 0), "%)"
appendInfoLine: "Skipped (close):    ", n_skipped_close
appendInfoLine: "Skipped (unvoiced): ", n_skipped_unvoiced
appendInfoLine: "Done -> ", output_wav$

# Cleanup
removeObject: snd_user, snd_ref, pitch_user, pitch_ref
removeObject: manipulation, pitch_tier, result
