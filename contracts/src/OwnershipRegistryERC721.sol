// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {ERC721} from "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";

/// @title OwnershipRegistryERC721
/// @notice PROJECT_DESIGN.md §8 Phase 4 "온체인 소유권 이전/ERC-721 승격" —
///         a real ERC-721 upgrade of OwnershipRegistry.sol, kept as a
///         separate contract rather than a rewrite of the original so the
///         MVP registry stays exactly as deployed and verified on Amoy.
///
///         Same content-hash-anchoring model as OwnershipRegistry.sol (only
///         a hash + doNotTrain flag on-chain, no image/metadata) and the
///         same relayer-sponsored registration path (PROJECT_DESIGN.md
///         §5-4) -- what changes is that "owner" is no longer a field this
///         contract tracks by hand. It's now `ownerOf(tokenId)`, standard
///         ERC-721 state, which is what makes marketplace/wallet
///         compatibility (OpenSea, MetaMask NFT display, `transferFrom`,
///         `approve`/`setApprovalForAll`) work for free instead of needing
///         a bespoke `transfer()` like the original's single-owner-only
///         path (see that contract's comment on why it didn't have one).
/// @dev Not deployed anywhere yet -- see contracts/README.md's "ERC-721
///      migration" section for the deploy/cutover plan. Migrating live data
///      from OwnershipRegistry.sol is a separate concern (re-registering
///      each contentHash here mints a *new* tokenId; there is no shared
///      numbering with the original contract).
contract OwnershipRegistryERC721 is ERC721, Ownable {
    struct Metadata {
        bytes32 contentHash;
        uint64 timestamp;
        bool doNotTrain;
    }

    /// @dev tokenId => on-chain metadata not covered by ERC721's own storage
    ///      (owner/approvals are ERC721's, not duplicated here).
    mapping(uint256 => Metadata) public metadata;

    /// @dev contentHash => tokenId, for O(1) duplicate checks and lookup by hash.
    mapping(bytes32 => uint256) public hashToToken;

    /// @dev Addresses allowed to call `registerFor` on behalf of a creator
    ///      (gas-sponsored registration via the platform relayer -- see
    ///      PROJECT_DESIGN.md §5-4). The relayer's own key is held in the
    ///      KMS, not in application code.
    mapping(address => bool) public relayers;

    uint256 public nextId;

    event Registered(uint256 indexed id, address indexed owner, bytes32 contentHash, bool doNotTrain);
    event RelayerUpdated(address indexed relayer, bool allowed);

    error AlreadyRegistered(bytes32 contentHash);
    error NotRelayer(address caller);
    error ZeroAddress();

    constructor(address initialOwner)
        ERC721("DONTAI Ownership Registry", "DONTAI")
        Ownable(initialOwner)
    {}

    modifier onlyRelayer() {
        if (!relayers[msg.sender]) revert NotRelayer(msg.sender);
        _;
    }

    /// @notice Self-service registration: caller pays their own gas and becomes the owner.
    function register(bytes32 contentHash, bool doNotTrain) external returns (uint256 id) {
        id = _register(msg.sender, contentHash, doNotTrain);
    }

    /// @notice Relayer-sponsored registration: the platform relayer submits the
    ///         transaction (and pays gas) on behalf of `owner`, who never needs
    ///         their own gas balance. Restricted to addresses the contract
    ///         owner has explicitly authorized via `setRelayer`.
    function registerFor(address owner, bytes32 contentHash, bool doNotTrain)
        external
        onlyRelayer
        returns (uint256 id)
    {
        if (owner == address(0)) revert ZeroAddress();
        id = _register(owner, contentHash, doNotTrain);
    }

    function _register(address owner, bytes32 contentHash, bool doNotTrain) private returns (uint256 id) {
        if (hashToToken[contentHash] != 0) revert AlreadyRegistered(contentHash);

        id = ++nextId;
        metadata[id] = Metadata({contentHash: contentHash, timestamp: uint64(block.timestamp), doNotTrain: doNotTrain});
        hashToToken[contentHash] = id;

        // _safeMint, not _mint -- registerFor lets a relayer mint to an
        // arbitrary owner address (a creator's custodial wallet) that this
        // contract has no other interaction with; if that address happens
        // to be a contract without ERC721Receiver support, this reverts the
        // registration instead of silently locking the token forever.
        _safeMint(owner, id);

        emit Registered(id, owner, contentHash, doNotTrain);
    }

    /// @notice Look up a registration by content hash (e.g. recomputed perceptual hash
    ///         of a suspected infringing image) to prove prior registration.
    function verify(bytes32 contentHash)
        external
        view
        returns (bool exists, address owner, uint64 timestamp, bool doNotTrain)
    {
        uint256 id = hashToToken[contentHash];
        if (id == 0) return (false, address(0), 0, false);

        Metadata storage m = metadata[id];
        // _ownerOf (not ownerOf) -- returns address(0) instead of reverting.
        // hashToToken[contentHash] != 0 already guarantees this token was
        // minted and never burned (this contract exposes no burn function),
        // so in practice this is always non-zero; using the non-reverting
        // form keeps `verify` a plain view call either way.
        return (true, _ownerOf(id), m.timestamp, m.doNotTrain);
    }

    /// @notice Authorize or revoke a relayer address. Only the contract owner
    ///         (platform admin) can call this.
    function setRelayer(address relayer, bool allowed) external onlyOwner {
        if (relayer == address(0)) revert ZeroAddress();
        relayers[relayer] = allowed;
        emit RelayerUpdated(relayer, allowed);
    }

    // Ownership transfer is standard ERC721 `transferFrom`/`safeTransferFrom`
    // (plus `approve`/`setApprovalForAll` for marketplace listings) --
    // inherited as-is, no override needed. This is the actual point of the
    // ERC-721 upgrade: OwnershipRegistry.sol's bespoke single-owner-only
    // `transfer()` doesn't support approvals or marketplace flows at all.
}
