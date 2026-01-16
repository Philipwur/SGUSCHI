#!/bin/bash
# ==================================================================================
# BindApptainerToSlurm.sh
# ==================================================================================
# Description:
#   This utility dynamically discovers Slurm binaries, libraries, configurations, 
#   and authentication sockets on the host system and sets up the necessary 
#   Apptainer (Singularity) bind paths. 
#
#   It enables a containerized application to submit jobs (sbatch), monitor queues 
#   (squeue), and interact with the cluster scheduler (scontrol/scancel) transparently,
#   resolving dependencies like Munge authentication and SSSD identity lookups.
#
# Usage:
#   source ./BindApptainerToSlurm.sh
#   apptainer exec my_container.sif sbatch ...
#
# Limitations:
#   1. OS Compatibility: Host and Container should share a similar glibc version if
#      deep library binding is required (though this script tries to be surgical).
#   2. Identity: If the container lacks the exact user database libraries, 
#      usernames might appear as UIDs (e.g., 1001 instead of 'user'), but functionality works.
#   3. Portability: While dynamic, extremely non-standard custom Slurm installs 
#      might require manual tweaks to the search paths.
#
# Useful if you want to run SGUCHI with an apptainer on a slurm cluster
# ==================================================================================

bind_apptainer_to_slurm() {
    local bind_args=()
    local added_paths=() # Cache to prevent duplicate binds

    # --- Helper: Add bind safely ---
    # Adds a path to the bind list if it exists.
    # Handles symlinks by binding both the link and the target to ensure validity.
    local add_bind
    add_bind() {
        local src="$1"
        local dest="${2:-$src}"
        local opts="${3:-ro}"
        
        # If source doesn't exist on host, skip it
        [ ! -e "$src" ] && return

        # Resolve fully qualified real path
        local real_src=$(readlink -f "$src")

        # Bind the real file/dir to its real location (if not already added)
        if [[ ! " ${added_paths[@]} " =~ " ${real_src} " ]]; then
            bind_args+=(--bind "$real_src:$real_src:$opts")
            added_paths+=("$real_src")
        fi

        # If it was a symlink, ALSO bind the symlink path itself
        # This keeps the "pointer" inside the container valid
        if [[ "$src" != "$real_src" ]]; then
             bind_args+=(--bind "$src:$src:$opts")
        fi
    }

    # ==============================================================================
    # 1. SLURM BINARIES
    # ==============================================================================
    local slurm_commands=(sbatch squeue scancel scontrol sinfo sacct)
    for cmd in "${slurm_commands[@]}"; do
        local cmd_path=$(which "$cmd" 2>/dev/null)
        [ -n "$cmd_path" ] && add_bind "$cmd_path" "$cmd_path" "ro"
    done

    # ==============================================================================
    # 2. SLURM LIBRARIES & PLUGINS
    # ==============================================================================
    # Use sbatch to find where libslurm is located
    local sbatch_path=$(which sbatch 2>/dev/null)
    local slurm_lib_dir=""
    if [ -n "$sbatch_path" ]; then
        local libslurm_path=$(ldd "$sbatch_path" | grep -E "libslurm(-|full|db)?" | head -n 1 | awk '{print $3}')
        if [ -n "$libslurm_path" ]; then
            slurm_lib_dir=$(dirname "$libslurm_path")
            add_bind "$slurm_lib_dir" "$slurm_lib_dir" "ro"
        fi
    fi

    # Find authentication plugins (munge) and their dependencies
    if [ -n "$slurm_lib_dir" ]; then
        # Look for auth_munge.so to find libmunge dependency
        local munge_plugin=$(find "$slurm_lib_dir" -name "auth_munge.so" 2>/dev/null | head -n 1)
        if [ -n "$munge_plugin" ]; then
            local munge_libs=$(ldd "$munge_plugin" | grep libmunge | awk '{print $3}')
            for lib in $munge_libs; do
                [ -f "$lib" ] && add_bind "$lib" "$lib" "ro"
            done
        fi
    fi

    # ==============================================================================
    # 3. SLURM CONFIGURATION
    # ==============================================================================
    local slurm_conf_path=""
    # Prefer env var, then ask scontrol, then check default
    if [ -n "$SLURM_CONF" ]; then
        slurm_conf_path="$SLURM_CONF"
    elif command -v scontrol &> /dev/null; then
        slurm_conf_path=$(scontrol show config 2>/dev/null | grep "SLURM_CONF" | awk '{print $3}')
    fi

    if [ -n "$slurm_conf_path" ]; then
        # Bind the config file and its directory
        add_bind "$slurm_conf_path" "$slurm_conf_path" "ro"
        add_bind "$(dirname "$slurm_conf_path")" "$(dirname "$slurm_conf_path")" "ro"
        
        # Deep Search: Parse "Include" directives in slurm.conf
        # If slurm.conf includes other files (e.g. gres.conf), we must bind them too.
        local includes=$(grep -i "^Include" "$slurm_conf_path" | awk '{print $2}')
        for inc in $includes; do
             add_bind "$inc" "$inc" "ro"
             add_bind "$(dirname "$inc")" "$(dirname "$inc")" "ro"
        done
        
        # EXPORT ENV: Tell Apptainer where to find the config
        export APPTAINER_ENV_SLURM_CONF="$slurm_conf_path"
    fi

    # ==============================================================================
    # 4. AUTHENTICATION & IDENTITY
    # ==============================================================================
    # Munge Sockets (RW access required)
    for munge_dir in /var/run/munge /run/munge /var/lib/munge; do
        if [ -d "$munge_dir" ]; then
            add_bind "$munge_dir" "$munge_dir" "rw"
        fi
    done

    # Basic Identity Files
    add_bind "/etc/passwd" "/etc/passwd" "ro"
    add_bind "/etc/group" "/etc/group" "ro"
    add_bind "/etc/nsswitch.conf" "/etc/nsswitch.conf" "ro"
    
    # SSSD Pipes and Cache (for networked user accounts)
    [ -d "/var/lib/sss/pipes" ] && add_bind "/var/lib/sss/pipes" "/var/lib/sss/pipes" "rw"
    [ -d "/var/lib/sss/mc" ]    && add_bind "/var/lib/sss/mc" "/var/lib/sss/mc" "ro"

    # Aggressive NSS Lib Search (libnss_sss, libnss_ldap, etc.)
    local nss_libs=$(find /usr/lib64 /lib64 -name "libnss_*.so*" 2>/dev/null)
    for lib in $nss_libs; do
        add_bind "$lib" "$lib" "ro"
    done

    # ==============================================================================
    # 5. EXPORT TO APPTAINER
    # ==============================================================================
    # We append our found binds to any existing binds in the environment variable
    local new_binds="${bind_args[*]}"
    
    # Remove array delimiters (space) and replace with comma if needed, 
    # though Apptainer accepts multiple --bind flags in APPTAINER_BIND if formatted correctly.
    # A cleaner approach for the ENV var is comma-separated paths: src:dest:opts
    
    # However, since we built bind_args with "--bind src:dest:opts", we need to strip "--bind "
    # to fit the APPTAINER_BIND environment variable format (src:dest:opts,src:dest:opts)
    
    local env_bind_str=""
    for arg in "${bind_args[@]}"; do
        if [[ "$arg" == "--bind" ]]; then continue; fi
        env_bind_str+="${arg},"
    done
    # Remove trailing comma
    env_bind_str=${env_bind_str%,}

    if [ -n "$APPTAINER_BIND" ]; then
        export APPTAINER_BIND="$APPTAINER_BIND,$env_bind_str"
    else
        export APPTAINER_BIND="$env_bind_str"
    fi
}

# Run the function immediately when sourced
bind_apptainer_to_slurm