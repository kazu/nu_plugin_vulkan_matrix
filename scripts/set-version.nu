def main [version: string] {
    let semver = '^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(-[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*)?(\+[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*)?$'
    if $version !~ $semver {
        error make {msg: 'Specify a SemVer version, for example V=0.1.0'}
    }
    let core = ($version | split row '+' | first)
    if ($core | str contains '-') {
        let pre = ($core | split row '-' | skip 1 | str join '-')
        if ($pre | split row '.' | any {|part| $part =~ '^[0-9]+$' and ($part | str length) > 1 and ($part | str starts-with '0')}) {
            error make {msg: 'Numeric prerelease identifiers cannot have leading zeroes'}
        }
    }

    let paths = [Cargo.toml validation/Cargo.toml Cargo.lock validation/Cargo.lock]
    let original = ($paths | each {|path| {path: $path, content: (open --raw $path)}})
    let root = ($original | where path == Cargo.toml | first | get content)
    let validation = ($original | where path == validation/Cargo.toml | first | get content)
    let old_requirement = ($validation | from toml | get dependencies.nu_plugin_vulkan_matrix.version)
    let dependency = $'nu_plugin_vulkan_matrix = { path = "..", version = "($old_requirement)" }'
    if not ($validation | str contains $dependency) {
        error make {msg: 'Cannot locate the validation dependency on the main package'}
    }
    let version_line = $'version = "($version)"'
    try {
        $root | str replace --regex '(?m)^version = "[^"]+"' $version_line | save --force Cargo.toml
        $validation
        | str replace --regex '(?m)^version = "[^"]+"' $version_line
        | str replace $dependency $'nu_plugin_vulkan_matrix = { path = "..", version = "($version)" }'
        | save --force validation/Cargo.toml
        for manifest in [Cargo.toml validation/Cargo.toml] {
            let result = (^cargo update --offline --workspace --manifest-path $manifest | complete)
            if $result.exit_code != 0 { error make {msg: $result.stderr} }
        }
    } catch {|err|
        for file in $original { $file.content | save --force $file.path }
        error make {msg: $err.msg}
    }
    print $'Version set to ($version); manifests and lockfiles are synchronized.'
}
