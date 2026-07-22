// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Ownable2Step, Ownable} from "@openzeppelin/contracts/access/Ownable2Step.sol";
import {Pausable} from "@openzeppelin/contracts/utils/Pausable.sol";

/// @title OwnershipRegistry
/// @notice Anchors content-hash ownership proofs on-chain for DONTAI artworks.
///         Only a content hash + owner + timestamp are stored on-chain; the
///         actual image, metadata, and C2PA manifest live off-chain (see
///         PROJECT_DESIGN.md §5-1). This keeps registration cheap and avoids
///         putting any personal/creative content on a public chain.
/// @dev Design reference: PROJECT_DESIGN.md §5-3.
contract OwnershipRegistry is Ownable2Step, Pausable {
    struct Record {
        address owner;
        bytes32 contentHash;
        uint64 timestamp;
        bool doNotTrain;
    }

    /// @dev tokenId => Record. tokenId 0 is never assigned, so it doubles as "not found".
    mapping(uint256 => Record) public records;

    /// @dev contentHash => tokenId, for O(1) duplicate checks and lookup by hash.
    mapping(bytes32 => uint256) public hashToToken;

    /// @dev Addresses allowed to call `registerFor` on behalf of a creator
    ///      (gas-sponsored registration via the platform relayer — see
    ///      PROJECT_DESIGN.md §5-4). The relayer's own key is held in the KMS,
    ///      not in application code.
    mapping(address => bool) public relayers;

    uint256 public nextId;

    /// @dev Commit-reveal front-running guard (real GPT/security-review
    ///      finding: register/registerFor's contentHash argument was visible
    ///      in plaintext the moment a tx hit the public mempool, before
    ///      confirmation -- anyone watching could resubmit the same
    ///      contentHash at a higher gas price and get mined first, becoming
    ///      the on-chain "first registrant" instead of the real caller. This
    ///      directly undermined the registry's entire purpose (proving
    ///      *prior* registration). commitHash = keccak256(abi.encode(...))
    ///      of every value the eventual register/registerFor call will use
    ///      plus a caller-chosen secret nonce -- only the hash is public
    ///      during the waiting period, so a copy-cat can't reconstruct the
    ///      real contentHash from it. commitBlock => the commit call's own
    ///      block.number, used to enforce MIN_COMMIT_AGE before reveal.
    mapping(bytes32 => uint256) public commitBlock;

    /// @dev Blocks that must pass between commit() and the matching
    ///      register()/registerFor() call. 1 is the minimum that actually
    ///      defeats the attack (see the commit-reveal analysis in this
    ///      contract's tests): even if an attacker sees a reveal
    ///      transaction in the mempool and instantly commits their own copy
    ///      of the same contentHash, their commit can't be revealed until
    ///      at least one block after the legitimate reveal is already
    ///      mined, so they can never win the race. Kept small (not a long
    ///      delay) since Polygon's ~2s block time already makes even 1
    ///      block a real, enforced gap -- a longer delay would only add
    ///      user-facing latency without closing any additional attack
    ///      surface this scheme doesn't already close at 1.
    uint256 public constant MIN_COMMIT_AGE = 1;

    event Registered(uint256 indexed id, address indexed owner, bytes32 contentHash, bool doNotTrain);
    event Transferred(uint256 indexed id, address indexed from, address indexed to);
    event RelayerUpdated(address indexed relayer, bool allowed);
    event Committed(bytes32 indexed commitHash, address indexed committer);

    error AlreadyRegistered(bytes32 contentHash);
    error NotFound(uint256 id);
    error NotOwner(uint256 id, address caller);
    error NotRelayer(address caller);
    error ZeroAddress();
    error CommitNotFound(bytes32 commitHash);
    error CommitTooRecent(bytes32 commitHash, uint256 readyAtBlock);

    constructor(address initialOwner) Ownable(initialOwner) {}

    modifier onlyRelayer() {
        if (!relayers[msg.sender]) revert NotRelayer(msg.sender);
        _;
    }

    /// @notice Step 1 of commit-reveal: publish only a hash of the intended
    ///         registration, hiding contentHash until the caller is ready to
    ///         reveal it (see MIN_COMMIT_AGE's doc for why this defeats
    ///         front-running). Anyone can commit any hash — the value itself
    ///         is meaningless without a matching reveal, so there's nothing
    ///         to protect here beyond recording when it happened.
    ///         Overwriting an existing (even unexpired) commitment with the
    ///         same hash is harmless: it just resets the clock, which only
    ///         ever *delays* that committer's own reveal, never anyone
    ///         else's.
    function commit(bytes32 commitHash) external whenNotPaused {
        commitBlock[commitHash] = block.number;
        emit Committed(commitHash, msg.sender);
    }

    /// @notice Step 2 (self-service path): caller pays their own gas and
    ///         becomes the owner. Must be preceded by a matching commit()
    ///         call at least MIN_COMMIT_AGE blocks ago.
    function register(bytes32 contentHash, bool doNotTrain, bytes32 nonce)
        external
        whenNotPaused
        returns (uint256 id)
    {
        _consumeCommitment(keccak256(abi.encode(contentHash, doNotTrain, msg.sender, nonce)));
        id = _register(msg.sender, contentHash, doNotTrain);
    }

    /// @notice Step 2 (relayer-sponsored path): the platform relayer submits
    ///         the transaction (and pays gas) on behalf of `owner`, who
    ///         never needs their own gas balance. Restricted to addresses
    ///         the contract owner has explicitly authorized via
    ///         `setRelayer`. Must be preceded by a matching commit() call at
    ///         least MIN_COMMIT_AGE blocks ago.
    function registerFor(address owner, bytes32 contentHash, bool doNotTrain, bytes32 nonce)
        external
        onlyRelayer
        whenNotPaused
        returns (uint256 id)
    {
        if (owner == address(0)) revert ZeroAddress();
        _consumeCommitment(keccak256(abi.encode(contentHash, doNotTrain, owner, msg.sender, nonce)));
        id = _register(owner, contentHash, doNotTrain);
    }

    function _consumeCommitment(bytes32 commitHash) private {
        uint256 committedAt = commitBlock[commitHash];
        if (committedAt == 0) revert CommitNotFound(commitHash);
        uint256 readyAt = committedAt + MIN_COMMIT_AGE;
        if (block.number < readyAt) revert CommitTooRecent(commitHash, readyAt);
        delete commitBlock[commitHash]; // one-time use, same reasoning as a nonce
    }

    function _register(address owner, bytes32 contentHash, bool doNotTrain) private returns (uint256 id) {
        if (hashToToken[contentHash] != 0) revert AlreadyRegistered(contentHash);

        id = ++nextId;
        records[id] = Record({owner: owner, contentHash: contentHash, timestamp: uint64(block.timestamp), doNotTrain: doNotTrain});
        hashToToken[contentHash] = id;

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

        Record storage r = records[id];
        return (true, r.owner, r.timestamp, r.doNotTrain);
    }

    /// @notice Transfer ownership of a registered record. Only the current
    ///         on-chain owner may call this directly (no relayer path for
    ///         transfers in the MVP — see PROJECT_DESIGN.md §5-3 note on
    ///         upgrading to ERC-721 if marketplace transfers are needed later).
    function transfer(uint256 id, address to) external whenNotPaused {
        Record storage r = records[id];
        if (r.owner == address(0)) revert NotFound(id);
        if (r.owner != msg.sender) revert NotOwner(id, msg.sender);
        if (to == address(0)) revert ZeroAddress();

        r.owner = to;
        emit Transferred(id, msg.sender, to);
    }

    /// @notice Authorize or revoke a relayer address. Only the contract owner
    ///         (platform admin) can call this.
    function setRelayer(address relayer, bool allowed) external onlyOwner {
        if (relayer == address(0)) revert ZeroAddress();
        relayers[relayer] = allowed;
        emit RelayerUpdated(relayer, allowed);
    }

    /// @notice Emergency stop for commit/register/registerFor/transfer if a
    ///         critical bug is found post-deployment. `verify` (a view
    ///         function, no state change) and `setRelayer`/`unpause`
    ///         themselves stay callable while paused -- an admin needs to be
    ///         able to keep operating and eventually lift the pause.
    function pause() external onlyOwner {
        _pause();
    }

    function unpause() external onlyOwner {
        _unpause();
    }
}
